"""Functions associated with the SMART model class
"""
import pickle
from collections import OrderedDict as odict
from dataclasses import dataclass
from decimal import Decimal
from itertools import chain
import logging

import dolfin as d
import numpy as np
import pandas
import petsc4py.PETSc as PETSc
import sympy as sym
from cached_property import cached_property
from sympy.parsing.sympy_parser import parse_expr
from tabulate import tabulate
from ufl.algorithms.ad import expand_derivatives
from ufl.form import sub_forms_by_domain

from .common import Stopwatch
from .common import sub
from .config import Config
from .mesh import ChildMesh, ParentMesh
from .model_assembly import (
    Compartment,
    CompartmentContainer,
    FluxContainer,
    Form,
    FormContainer,
    Parameter,
    ParameterContainer,
    Reaction,
    ReactionContainer,
    Species,
    SpeciesContainer,
    empty_sbmodel,
    ParameterType,
)
from .solvers import smartSNESProblem
from .units import unit


logger = logging.getLogger(__name__)


@dataclass
class Model:
    """SMART model class: consists of parameters,
    species, compartments, reactions.

    Args:
        pc: Parameter container, initialized in model_assembly
        sc: Species container, initialized in model_assembly
        cc: Compartment container, initialized in model_assembly
        rc: Reaction container, initialized in model_assembly
        config: configuration object initialized in config.py
        parent_mesh: ParentMesh object initialized in mesh.py
        name (optional): string specifying a name for your model

    .. note::
        Given a parent mesh
        and solver, the system can be solved for spatiotemporal dynamics.
        Object is initialized by calling:

        .. code:: python

            model = model.Model(pc, sc, cc, rc, config, parent_mesh)
    """

    pc: ParameterContainer
    sc: SpeciesContainer
    cc: CompartmentContainer
    rc: ReactionContainer
    config: Config
    # solver_system: smart.solvers.SolverSystem
    parent_mesh: ParentMesh
    name: str = ""

    def to_dict(self):
        """Convert model information to Dict"""
        parameters = self.pc.to_dicts()
        species = self.sc.to_dicts()
        compartments = self.cc.to_dicts()
        reactions = self.rc.to_dicts()
        return {
            "name": self.name,
            "parameters": parameters,
            "species": species,
            "compartments": compartments,
            "reactions": reactions,
            "parent_mesh_filename": self.parent_mesh.mesh_filename,
            "parent_mesh_filetype": self.parent_mesh.mesh_filetype,
            "config": self.config.__dict__,
        }

    @classmethod
    def from_dict(cls, input_dict):
        """Read model information from Dict"""
        pc, sc, cc, rc = empty_sbmodel()
        pc.add([Parameter.from_dict(parameter) for parameter in input_dict["parameters"]])
        sc.add([Species.from_dict(species) for species in input_dict["species"]])
        cc.add([Compartment.from_dict(compartment) for compartment in input_dict["compartments"]])
        rc.add([Reaction.from_dict(reaction) for reaction in input_dict["reactions"]])
        config = Config()
        config.__dict__ = input_dict["config"]
        parent_mesh = ParentMesh(
            input_dict["parent_mesh_filename"], input_dict["parent_mesh_filetype"]
        )
        return cls(pc, sc, cc, rc, config, parent_mesh, input_dict["name"])

    def to_pickle(self, filename):
        """Save model information to file by pickling"""
        with open(filename, "wb") as f:
            pickle.dump(self.to_dict(), f)

    @classmethod
    def from_pickle(cls, filename):
        """Read model information from file by unpickling"""
        with open(filename, "rb") as f:
            input_dict = pickle.load(f)
        return cls.from_dict(input_dict)

    def __post_init__(self):
        """Initialize solver-related parameters, timers, and MPI."""
        # # Check that solver_system is valid

        # FunctionSpaces, Functions, etc
        self.V = dict()
        self.u = dict()

        self.fc = FluxContainer()

        # Solver related parameters
        self.idx = 0
        self.idx_nl = list()
        self.idx_l = list()
        self.reset_dt = False

        self._failed_to_converge = False
        # idx, idx_nl, idx_l, t, dt, residuals, reason
        self.failed_solves = list()
        # list of dicts
        self.residuals = list()

        # Timers
        stopwatch_names = [
            "Total time step",
            "Total simulation",
            "loading mesh",
            "Total initialization",
            "snes all",
            "snes total solve",
            "snes total assemble",
            "snes jacobian assemble",
            "snes residual assemble",
            "snes initialize zero matrices",
        ]

        # nicer printing for timers
        print_buffer = max([len(stopwatch_name) for stopwatch_name in stopwatch_names])
        self.stopwatches = {
            stopwatch_name: Stopwatch(stopwatch_name, print_buffer=print_buffer)
            for stopwatch_name in stopwatch_names
        }

        # Functional forms
        self.forms = FormContainer()

        # MPI - define to be consistent with mesh
        self.mpi_comm_world = self.parent_mesh.mpi_comm
        self.mpi_rank = self.mpi_comm_world.rank
        self.mpi_size = self.mpi_comm_world.size
        self.mpi_root = 0

        if self.mpi_size > 1:
            logger.info(
                f"CPU {self.mpi_rank}: Model '{self.name}' "
                f"has been parallelized (size={self.mpi_size}).",
                extra=dict(format_type="log_urgent"),
            )

    @property
    def mpi_am_i_root(self):
        """Returns True if current process is root"""
        return self.mpi_rank == self.mpi_root

    @property
    def child_meshes(self):
        """Returns all child meshes in current parent mesh"""
        return self.parent_mesh.child_meshes

    @cached_property
    def min_dim(self):
        """Returns minimum dimension in current model"""
        dim = min([comp.dimensionality for comp in self.cc])
        self.parent_mesh.min_dim = dim
        return dim

    @cached_property
    def max_dim(self):
        """Returns maximum dimension in current model"""
        dim = max([comp.dimensionality for comp in self.cc])
        self.max_dim = dim
        self.parent_mesh.max_dim = dim
        return dim

    def initialize(self, initialize_solver=True):
        """Main model initialization function, split into 5 subfunctions"""

        # Solver related parameters
        self._base_t = Decimal("0." + (self.config.solver["time_precision"] - 1) * "0" + "1")
        self.t = self.rounded_decimal(0.0)
        self.dt = self.rounded_decimal(self.config.solver["initial_dt"])
        self.final_t = self.rounded_decimal(self.config.solver["final_t"])
        assert self.config.solver["time_precision"] in range(1, 30)

        self.T = d.Constant(self.t)
        self.dT = d.Constant(self.dt)
        self.tvec = [self.t]
        self.dtvec = [self.dt]

        self._init_1()
        self._init_2()
        self._init_3()
        self._init_4()
        self._init_5(initialize_solver)

        logger.debug("Model finished initialization!", extra=dict(format_type="title"))
        if self.config.flags["print_verbose_info"]:
            self.pc.print()
            self.sc.print()
            self.cc.print()
            self.print_meshes()
            self.rc.print()

    def _init_1(self):
        """Checking validity of model"""
        logger.debug("Checking validity of model (step 1 of ZZ)", extra=dict(format_type="title"))
        self._init_1_1_check_mesh_dimensionality()
        self._init_1_2_check_namespace_conflicts()
        self._init_1_3_check_parameter_dimensionality()
        logger.debug(
            "Step 1 of initialization completed successfully!",
            extra=dict(text_color="magenta"),
        )

    def _init_2(self):
        """Cross-container dependent initializations
        (requires information from multiple containers)"""
        logger.debug(
            "Cross-Container Dependent Initializations (step 2 of ZZ)",
            extra=dict(
                format_type="title",
            ),
        )
        self._init_2_1_reactions_to_symbolic_strings()
        self._init_2_2_check_reaction_validity()
        self._init_2_3_link_reaction_properties()
        self._init_2_4_check_for_unused_parameters_species_compartments()
        self._init_2_5_link_compartments_to_species()
        self._init_2_6_link_species_to_compartments()
        self._init_2_7_get_species_compartment_indices()
        logger.debug(
            "Step 2 of initialization completed " "successfully!",
            extra=dict(text_color="magenta"),
        )

    def _init_3(self):
        """Mesh-related initializations"""
        logger.debug(
            "Mesh-related Initializations (step 3 of ZZ)",
            extra=dict(format_type="title"),
        )
        self._init_3_1_define_child_meshes()
        self._init_3_2_read_parent_mesh_functions_from_file()
        self._init_3_3_extract_submeshes()
        self._init_3_4_build_submesh_mappings()
        logger.debug("DEBUGGING 3_5, 3_6 (mesh intersections)", extra=dict(format_type="warning"))
        # self._init_3_5_get_child_mesh_intersections()
        # self._init_3_6_get_intersection_submeshes()
        self._init_3_7_get_integration_measures()
        logger.debug(
            "Step 3 of initialization completed successfully!",
            extra=dict(format_type="log_important"),
        )

    def _init_4(self):
        """Dolfin function initializations"""
        logger.debug("Dolfin Initializations (step 4 of ZZ)", extra=dict(format_type="title"))
        self._init_4_0_initialize_dolfin_parameters()
        self._init_4_1_get_active_compartments()
        self._init_4_2_define_dolfin_function_spaces()
        self._init_4_3_define_dolfin_functions()
        self._init_4_4_get_species_u_v_V_dofmaps()
        self._init_4_5_name_functions()
        self._init_4_6_check_dolfin_function_validity()
        self._init_4_7_set_initial_conditions()

    def _init_5(self, initialize_solver):
        """Convert reactions to fluxes and define variational form.
        If initialize_solver is true, also initialize solver.
        """
        logger.debug(
            "Dolfin fluxes, forms, and problems+solvers (step 5 of ZZ)",
            extra=dict(
                format_type="title",
            ),
        )
        self._init_5_1_reactions_to_fluxes()
        self._init_5_2_create_variational_forms()
        if initialize_solver:
            self.initialize_discrete_variational_problem_and_solver()

    # Step 1 - Checking model validity

    def _init_1_1_check_mesh_dimensionality(self):
        """Dimensionality checks:

        * Error if max_dim - min_dim is greater than 1
        * Error if max_dim (from compartments) is greater than dimensionality of parent mesh
        * Warning if max_dim is less than dimensionality of parent mesh
        """
        logger.debug(
            "Check that mesh/compartment dimensionalities match",
            extra=dict(format_type="log"),
        )

        if (self.max_dim - self.min_dim) not in [0, 1]:
            raise ValueError(
                "(Highest mesh dimension - smallest mesh dimension) must be either 0 or 1."
            )

        if self.max_dim > self.parent_mesh.dimensionality:
            raise ValueError(
                "Maximum dimension of a compartment is "
                "higher than the topological dimension of parent mesh."
            )
        # This is possible to simulate but could be unintended.
        # (e.g. if you have a 3d mesh you could choose to only simulate on the surface)
        if self.max_dim != self.parent_mesh.dimensionality:
            logger.warning(
                "Parent mesh has geometric dimension: "
                f"{self.parent_mesh.dimensionality} which"
                f" is not the same as the maximum compartment dimension: {self.max_dim}.",
                extra=dict(format_type="warning"),
            )

        for compartment in self.cc:
            compartment.is_volume = compartment.dimensionality == self.max_dim

    def _init_1_2_check_namespace_conflicts(self):
        """Namespace checks:

        * no repeated names of parameters/species/compartments/reactions
        * no use of x[0], x[1], x[2], t, or unit_scale_factor as names
        * no two compartments share the same marker number
        * no compartment has a marker value of 0
        """
        logger.debug("Checking for namespace conflicts", extra=dict(format_type="log"))
        self._all_keys = set()
        containers = [self.pc, self.sc, self.cc, self.rc]
        for keys in [c.keys for c in containers]:
            self._all_keys = self._all_keys.union(keys)
        if sum([c.size for c in containers]) != len(self._all_keys):
            raise ValueError(
                "Model has a namespace conflict. There are "
                "two parameters/species/compartments/reactions with the same name."
            )

        protected_names = {"x[0]", "x[1]", "x[2]", "t", "unit_scale_factor"}
        # Protect the variable names 'x[0]', 'x[1]', 'x[2]' and 't' because they are
        # used for spatial dimensions and time
        if not protected_names.isdisjoint(self._all_keys):
            raise ValueError(
                "An object is using a protected variable name "
                "('x[0]', 'x[1]', 'x[2]', 't', or 'unit_scale_factor'). Please change the name."
            )

        # Make sure there are no overlapping markers or markers with value 0
        self._all_markers = set()
        for compartment in self.cc.values:
            if isinstance(compartment.cell_marker, list):
                marker_list = compartment.cell_marker
            else:
                marker_list = [compartment.cell_marker]

            for marker in marker_list:
                if marker in self._all_markers:
                    raise ValueError(f"Two compartments have the same marker: {marker}")
                elif marker == 0:
                    raise ValueError("Marker cannot have the value 0")
                else:
                    self._all_markers.add(marker)

    def _init_1_3_check_parameter_dimensionality(self):
        """Check no objects reference a higher spatial dimension than max_dim"""
        if "x[2]" in self._all_keys and self.max_dim < 3:
            raise ValueError(
                "An object has the variable name 'x[2]' but there "
                "are less than 3 spatial dimensions."
            )
        if "x[1]" in self._all_keys and self.max_dim < 2:
            raise ValueError(
                "An object has the variable name 'x[1]' but there are "
                "less than 2 spatial dimensions."
            )

    # Step 2 - Cross-container Dependent Initialization

    def _init_2_1_reactions_to_symbolic_strings(self):
        """Turn all reactions into unsigned symbolic flux strings"""
        logger.debug(
            "Turning reactions into unsigned symbolic flux strings",
            extra=dict(format_type="log"),
        )

        for reaction in self.rc:
            # Mass action has a forward and reverse flux
            if reaction.reaction_type == "mass_action":
                reaction.eqn_f_str = reaction.param_map["on"]
                for sp_name in reaction.lhs:
                    reaction.eqn_f_str += "*" + sp_name
                # rxn.eqn_f = parse_expr(rxn_sym_str)

                reaction.eqn_r_str = reaction.param_map["off"]
                for sp_name in reaction.rhs:
                    reaction.eqn_r_str += "*" + sp_name
                # rxn.eqn_r = parse_expr(rxn_sym_str)

            elif reaction.reaction_type == "mass_action_forward":
                reaction.eqn_f_str = reaction.param_map["on"]
                for sp_name in reaction.lhs:
                    reaction.eqn_f_str += "*" + sp_name
                # reaction.eqn_f = parse_expr(rxn_sym_str)

            # Custom reaction
            elif reaction.reaction_type in self.config.reaction_database.keys():
                reaction.eqn_f_str = reaction._parse_custom_reaction(
                    self.config.reaction_database[reaction.reaction_type]
                )

            # pre-defined equation string
            elif reaction.eqn_f_str or reaction.eqn_r_str:
                if reaction.eqn_f_str:
                    reaction.eqn_f_str = reaction._parse_custom_reaction(reaction.eqn_f_str)
                if reaction.eqn_r_str:
                    reaction.eqn_r_str = reaction._parse_custom_reaction(reaction.eqn_r_str)

            else:
                raise ValueError(
                    "Reaction %s does not seem to have an associated equation" % reaction.name
                )

    def _init_2_2_check_reaction_validity(self):
        """Confirms that all reactions have parameters/species defined"""
        logger.debug(
            "Make sure all reactions have parameters/species defined",
            extra=dict(format_type="log"),
        )
        # Make sure all reactions have parameters/species defined
        for reaction in self.rc:
            for eqn_str in [reaction.eqn_f_str, reaction.eqn_r_str]:
                if not eqn_str:
                    continue
                eqn = parse_expr(eqn_str)
                var_set = {str(x) for x in eqn.free_symbols}
                param_set = var_set.intersection(self.pc.keys)
                species_set = var_set.intersection(self.sc.keys)
                if len(param_set) + len(species_set) != len(var_set):
                    diff_set = var_set.difference(param_set.union(species_set))
                    raise NameError(
                        f"Reaction {reaction.name} refers to a parameter or "
                        f"species ({diff_set}) that is not in the model."
                    )

    def _init_2_3_link_reaction_properties(self):
        """Link parameters, species, and compartments to reactions"""
        logger.debug(
            "Linking parameters, species, and compartments to reactions",
            extra=dict(format_type="log"),
        )

        for reaction in self.rc:
            reaction.parameters = {
                param_name: self.pc[param_name] for param_name in reaction.param_map.values()
            }
            reaction.species = {
                species_name: self.sc[species_name]
                for species_name in reaction.species_map.values()
            }
            compartment_names = [species.compartment_name for species in reaction.species.values()]
            if reaction.explicit_restriction_to_domain:
                compartment_names.append(reaction.explicit_restriction_to_domain)
            reaction.compartments = {
                compartment_name: self.cc[compartment_name]
                for compartment_name in compartment_names
            }
            # number of parameters, species, and compartments
            reaction.num_species = len(reaction.species)

            is_volume = [c.is_volume for c in reaction.compartments.values()]
            if len(is_volume) == 1:
                if is_volume[0]:
                    reaction.topology = "volume"
                else:
                    reaction.topology = "surface"
            elif len(is_volume) == 2:
                if all(is_volume):
                    raise NotImplementedError(
                        f"Reaction {reaction.name} has two volumes - "
                        "the adjoining surface must be specified "
                        "using reaction.explicit_restriction_to_domain"
                    )
                    # reaction.topology = 'volume_volume'
                elif not any(is_volume):
                    raise Exception(
                        f"Reaction {reaction.name} involves two surfaces. " "This is not supported."
                    )
                else:
                    reaction.topology = "volume_surface"
            elif len(is_volume) == 3:
                if sum(is_volume) == 3:
                    raise Exception(
                        f"Reaction {reaction.name} involves three volumes. "
                        "This is not supported."
                    )
                elif sum(is_volume) < 2:
                    raise Exception(
                        f"Reaction {reaction.name} involves two or more surfaces. "
                        "This is not supported."
                    )
                else:
                    reaction.topology = "volume_surface_volume"
            else:
                raise ValueError("Number of compartments involved in a flux must be in [1,2,3]!")

    def _init_2_4_check_for_unused_parameters_species_compartments(self):
        """
        If :code:`self.config.flags["allow_unused_components"]` is true,
        then unused objects are removed
        """
        logger.debug(
            "Checking for unused parameters, species, or compartments",
            extra=dict(format_type="log"),
        )

        all_parameters = set(chain.from_iterable([r.parameters for r in self.rc]))
        all_species = set(chain.from_iterable([r.species for r in self.rc]))
        all_compartments = set(chain.from_iterable([r.compartments for r in self.rc]))
        if all_parameters != set(self.pc.keys):
            print_str = (
                f"Parameter(s), {set(self.pc.keys).difference(all_parameters)}, "
                "are unused in any reactions."
            )
            if self.config.flags["allow_unused_components"]:
                for parameter in set(self.pc.keys).difference(all_parameters):
                    self.pc.remove(parameter)
                logger.info(print_str, extra=dict(format_type="log_urgent"))
                logger.info(
                    "Removing unused parameter(s) from model!",
                    extra=dict(format_type="log_urgent"),
                )
            else:
                raise ValueError(print_str)
        if all_species != set(self.sc.keys):
            print_str = (
                f"Species, {set(self.sc.keys).difference(all_species)}, "
                "are unused in any reactions."
            )
            if self.config.flags["allow_unused_components"]:
                for species in set(self.sc.keys).difference(all_species):
                    self.sc.remove(species)
                logger.info(print_str, extra=dict(format_type="log_urgent"))
                logger.info(
                    "Removing unused species(s) from model!",
                    extra=dict(format_type="log_urgent"),
                )
            else:
                raise ValueError(print_str)
        if all_compartments != set(self.cc.keys):
            print_str = (
                f"Compartment(s), {set(self.cc.keys).difference(all_compartments)}, "
                "are unused in any reactions."
            )
            if self.config.flags["allow_unused_components"]:
                for compartment in set(self.cc.keys).difference(all_compartments):
                    self.cc.remove(compartment)
                logger.info(print_str, extra=dict(format_type="log_urgent"))
                logger.info(
                    "Removing unused compartment(s) from model!",
                    extra=dict(format_type="log_urgent"),
                )
            else:
                raise ValueError(print_str)

    def _init_2_5_link_compartments_to_species(self):
        """Linking compartments and compartment dimensionality to species"""
        logger.debug(
            "Linking compartments and compartment dimensionality to species",
            extra=dict(format_type="log"),
        )
        for species in self.sc:
            species.compartment = self.cc[species.compartment_name]
            species.dimensionality = self.cc[species.compartment_name].dimensionality

    def _init_2_6_link_species_to_compartments(self):
        """Links species to compartments - a species is considered to be
        "in a compartment" if it is involved in a reaction there
        """
        logger.debug("Linking species to compartments", extra=dict(format_type="log"))
        # An species is considered to be "in a compartment" if it is
        # involved in a reaction there
        for species in self.sc:
            species.compartment.species.update({species.name: species})
        for compartment in self.cc:
            compartment.num_species = len(compartment.species)

    def _init_2_7_get_species_compartment_indices(self):
        """Store dof indices for species for each compartment"""
        logger.debug(
            "Getting indices for species for each compartment",
            extra=dict(format_type="log"),
        )
        for compartment in self.cc:
            index = 0
            for species in list(compartment.species.values()):
                species.dof_index = index
                index += 1

    # Step 3 - Mesh Initializations
    def _init_3_1_define_child_meshes(self):
        """Initialize all ChildMesh objects"""
        logger.debug("Defining child meshes", extra=dict(format_type="log"))
        # Check that there is a parent mesh loaded
        if not isinstance(self.parent_mesh, ParentMesh):
            raise ValueError("There is no parent mesh.")

        # Define child meshes
        for comp in self.cc:
            comp.mesh = ChildMesh(self.parent_mesh, comp)

        # Check validity (one child mesh for each compartment)
        assert len(self.child_meshes) == self.cc.size

    def _init_3_2_read_parent_mesh_functions_from_file(self):
        """Reads mesh functions from an .xml or .hdf5 file.
        If told that a child mesh consists of a list of markers,
        creates a separate mesh function
        for the list and for the combined list (can be used for post-processing)
        """
        logger.debug("Defining parent mesh functions", extra=dict(format_type="log"))
        self.parent_mesh.read_parent_mesh_functions_from_file()

    def _init_3_3_extract_submeshes(self):
        """Use :code:`dolfin.MeshView.create()` to extract submeshes"""
        logger.debug("Extracting submeshes using MeshView", extra=dict(format_type="log"))
        # Loop through child meshes and extract submeshes
        for child_mesh in self.child_meshes.values():
            child_mesh.extract_submesh()

    def _init_3_4_build_submesh_mappings(self):
        """Build MeshView mappings between all child mesh pairs"""
        logger.debug(
            "Building MeshView mappings between all child mesh pairs",
            extra=dict(format_type="log"),
        )

        # not creating maps with build_mapping() will lead to a
        # segfault when trying to assemble
        # we also suppress the c++ output because it prints a lot of 'errors'
        # even though the mapping was successfully built
        # (when (surface intersect volume) != surface)
        # with common._stdout_redirected():
        for child_mesh in self.parent_mesh.child_surface_meshes:
            for sibling_volume_mesh in self.parent_mesh.child_volume_meshes:
                if hasattr(child_mesh.compartment, "nonadjacent_compartment_list"):
                    if (
                        sibling_volume_mesh.compartment.name
                        in child_mesh.compartment.nonadjacent_compartment_list
                    ):
                        logger.debug(
                            "Skipping mapping between {} and {}".format(
                                child_mesh.compartment.name,
                                sibling_volume_mesh.compartment.name,
                            ),
                            extra=dict(format_type="log"),
                        )
                        continue
                child_mesh.dolfin_mesh.build_mapping(sibling_volume_mesh.dolfin_mesh)

    def _init_3_7_get_integration_measures(self):
        """Get integration measures for parent mesh and child meshes"""
        logger.debug(
            "Getting integration measures for parent mesh and child meshes",
            extra=dict(format_type="log"),
        )
        for mesh in self.parent_mesh.all_meshes.values():
            mesh.get_integration_measures()

    def _init_4_0_initialize_dolfin_parameters(self):
        """Create dolfin objects for each parameter.
        If we were to re-create dolfin constants
        each time, FEniCS will recompile the form
        so we only define them once here.
        Therefore, once the model is initialized, changing a parameter's
        value does not change the underlying dolfin object.
        Values must be assigned via :code:`parameter.dolfin_constant.assign()`
        """
        # Create a dolfin.Constant() for constant parameters
        for parameter in self.pc.values:
            if parameter.type == ParameterType.constant:
                parameter.dolfin_constant = d.Constant(parameter.value)
            elif parameter.type == ParameterType.expression and not parameter.use_preintegration:
                parameter.dolfin_expression = d.Expression(
                    sym.printing.ccode(parameter.sym_expr), t=self.T, degree=1
                )
            elif parameter.type == ParameterType.expression and parameter.use_preintegration:
                parameter.dolfin_constant = d.Constant(parameter.value)
            elif parameter.type == ParameterType.from_file:
                parameter.dolfin_constant = d.Constant(parameter.value)

    def _init_4_1_get_active_compartments(self):
        """Arrange the compartments based on the number of degrees of freedom they have
        (We want to have the highest number of dofs first)
        """
        # addressing https://github.com/justinlaughlin/smart/issues/36
        self._active_compartments = list(self.cc.Dict.values())
        self._all_compartments = list(self.cc.Dict.values())
        it_idx = 0
        rm_idx = np.array([])
        for compartment in self._active_compartments:
            if compartment.num_species < 1:
                rm_idx = np.append(rm_idx, it_idx)
            it_idx = it_idx + 1
        for i in range(len(rm_idx)):
            rm_cur = int(rm_idx[i])
            del self._active_compartments[rm_cur]
            rm_idx = rm_idx - 1  # adjust indices to compensate after deleting
        for idx, compartment in enumerate(self._active_compartments):
            compartment.dof_index = idx

    # Step 4 - Dolfin Functions

    def _init_4_2_define_dolfin_function_spaces(self):
        """Define dolfin function spaces for compartments
        For each compartment, a P1 (linear shape functions for a tri or tet element)
        vector function space is used, with the vector dimensionality given
        by the number of species in the compartment.
        The mixed function space W is the product space of all vector function
        spaces of individual compartments
        """
        logger.debug(
            "Defining dolfin function spaces for compartments",
            extra=dict(format_type="log"),
        )
        # Aliases
        max_compartment_name = max([len(compartment_name) for compartment_name in self.cc.keys])

        # Make the individual function spaces (per compartment)
        for compartment in self._active_compartments:
            # Aliases
            logger.debug(
                f"Defining function space for {compartment.name}"
                f"{' '*(max_compartment_name-len(compartment.name))} "
                f"(dim: {compartment.dimensionality}, "
                f"species: {compartment.num_species}, dofs: {compartment.num_dofs})",
                extra=dict(format_type="log"),
            )

            if compartment.num_species > 1:
                compartment.V = d.VectorFunctionSpace(
                    self.child_meshes[compartment.name].dolfin_mesh,
                    "P",
                    1,
                    dim=compartment.num_species,
                )
            else:
                compartment.V = d.FunctionSpace(
                    self.child_meshes[compartment.name].dolfin_mesh, "P", 1
                )

        self.V = [compartment.V for compartment in self._active_compartments]
        # Make the MixedFunctionSpace
        self.W = d.MixedFunctionSpace(*self.V)

    def _init_4_3_define_dolfin_functions(self):
        """Define test functions and trial functions in the mixed function space

        .. note::
            Some notes on functions in VectorFunctionSpaces:
            :code:`u.sub(idx)` gives us a shallow copy to the idx sub-function of
            :code:`u`.
            This is useful when we want to look at function values
            :code:`u.split()[idx]` is equivalent to :code:`u.sub(idx)`

            :code:`d.split(u)[idx]` (same thing as ufl.split(u)) gives us
            ufl objects (:code:`ufl.indexed.Indexed`)
            that can be used in variational formulations.

            :code:`u.sub(idx)` can be used in a variational formulation but
            it won't behave properly.

            For TestFunctions/TrialFunctions, we will always get a
            :code:`ufl.indexed.Indexed` object

            .. code python
                # type(v) == d.function.argument.Argument
                v = d.TestFunction(V)
                v[0] == d.split(v)[0]
                # type(v_) == tuple(ufl.indexed.Indexed, ufl.indexed.Indexed)
                v_ = d.TestFunctions(V)
                v_[0] == v[0] == d.split(v)[0]

            For a MixedFunctionSpace e.g. :code:`W=d.MixedFunctionSpace(*[V1,V2])`
            we can use :code:`sub()` to get the subfunctions
        """
        logger.debug("Defining dolfin functions", extra=dict(format_type="log"))
        # dolfin functions created from MixedFunctionSpace
        self.u["u"] = d.Function(self.W)
        # self.u['k'] = d.Function(self.W)
        self.u["n"] = d.Function(self.W)

        # Trial and test functions
        self.ut = d.TrialFunctions(self.W)  # list of TrialFunctions
        self.v = d.TestFunctions(self.W)  # list of TestFunctions

        # Create references in compartments to the subfunctions
        for compartment in self._active_compartments:
            # alias
            cidx = compartment.dof_index

            # functions
            for key, func in self.u.items():
                # compartment.u[key] = sub(func,cidx) #func.sub(cidx)
                # u is a function from a MixedFunctionSpace so u.sub()
                # is appropriate here
                compartment.u[key] = sub(func, cidx)  # func.sub(cidx)
                if compartment.u[key].num_sub_spaces() > 1:
                    # _usplit are the actual functions we use to construct
                    # variational forms
                    compartment._usplit[key] = d.split(compartment.u[key])
                else:
                    compartment._usplit[key] = (compartment.u[key],)  # one element tuple

            # since we are using TrialFunctions() and TestFunctions()
            # this is the proper
            # syntax even if there is just one compartment
            compartment.ut = sub(self.ut, cidx)  # self.ut[cidx]
            compartment.v = sub(self.v, cidx)  # self.v[cidx]

        # save these in model
        self._usplit = [c._usplit["u"] for c in self._active_compartments]

    def _init_4_4_get_species_u_v_V_dofmaps(self):
        """Extract subfunctions, function spaces, dofmap for each species"""
        logger.debug(
            "Extracting subfunctions/function spaces/dofmap for each species",
            extra=dict(format_type="log"),
        )
        for compartment in self._active_compartments:
            # loop through species and add the name/index
            for species in compartment.species.values():
                species.V = sub(compartment.V, species.dof_index)
                species.v = sub(compartment.v, species.dof_index)
                species.dof_map = self.dolfin_get_dof_indices(species)  # species.V.dofmap().dofs()

                for key in compartment.u.keys():
                    # compartment.u[key].sub(species.dof_index)
                    species.u[key] = sub(compartment.u[key], species.dof_index)
                    # compartment.u[key].sub(species.dof_index)
                    species._usplit[key] = sub(compartment._usplit[key], species.dof_index)
                species.ut = sub(compartment.ut, species.dof_index)

    def _init_4_5_name_functions(self):
        """Assign function names based on compartment name, species index, and species name"""
        logger.debug("Naming functions and subfunctions", extra=dict(format_type="log"))
        for compartment in self._active_compartments:
            # name of the compartment function
            for key in self.u.keys():
                compartment.u[key].rename(f"{compartment.name}_{key}", "")
                # loop through species and add the name/index
                for species in compartment.species.values():
                    sidx = species.dof_index
                    if compartment.num_species > 1:
                        species.u[key].rename(f"{compartment.name}_{sidx}_{species.name}_{key}", "")

    def _init_4_6_check_dolfin_function_validity(self):
        """
        Checks to confirm that dolfin functions were created correctly:

        * function size in compartment == dof in that compartment
        * number of sub-spaces in compartment == number of species in compartment
        * function space of compartment matches sub-space of W
        """
        logger.debug(
            "Checking that dolfin functions were created correctly",
            extra=dict(format_type="log"),
        )
        # sanity check
        for compartment in self._active_compartments:  # self.cc:
            idx = compartment.dof_index
            compartment.num_dofs
            compartment.num_dofs_local
            # function size == dofs
            if self.mpi_size == 1:
                assert compartment.u["u"].vector().size() == compartment._num_dofs
            if self.mpi_size >= 1:
                assert compartment.u["u"].vector().get_local().size == compartment._num_dofs_local

            # number of sub spaces == number of species
            if compartment.num_species == 1:
                for ukey in compartment.u.keys():
                    assert compartment.u[ukey].num_sub_spaces() == 0
            else:
                for ukey in compartment.u.keys():
                    assert compartment.u[ukey].num_sub_spaces() == compartment.num_species

            # function space matches W.sub(idx)
            for func in list(compartment.u.values()) + [compartment.v]:
                assert func.function_space().id() == self.W.sub_space(idx).id()

    def _init_4_7_set_initial_conditions(self):
        """
        Sets the function values to initial conditions,
        either based on an expression given by a string
        or a numerical value (float)
        (see model.dolfin_set_function_values for further details)
        """
        logger.debug("Set function values to initial conditions", extra=dict(format_type="log"))
        for species in self.sc:
            for ukey in species.u.keys():
                if isinstance(species.initial_condition, float) or not isinstance(
                    species.initial_condition, str
                ):
                    self.dolfin_set_function_values(species, ukey, species.initial_condition)
                else:
                    self.dolfin_set_function_values(
                        species, ukey, species.initial_condition_expression
                    )

    def _init_5_1_reactions_to_fluxes(self):
        """Convert reactions to flux objects"""
        logger.debug("Convert reactions to flux objects", extra=dict(format_type="log"))
        for reaction in self.rc:
            reaction.reaction_to_fluxes()
            self.fc.add(reaction.fluxes)

    def _init_5_2_create_variational_forms(self):
        """
        Setup the variational forms in dolfin
        Forms:

        .. code:: python

            F(u;v) =    Muform    + Munform   +    Dform     +     Rform              = 0
                    linear wrt u      (v)      linear wrt u  possibly nonlinear wrt u
        """
        logger.debug("Creating functional forms", extra=dict(format_type="log"))

        # default dictionary (linear w.r.t all compartment functions)
        linear_wrt_comp = {k: True for k in self.cc.keys}
        # nonlinear_wrt_comp = {k: False for k in self.cc.keys}

        # reactive terms
        for flux in self.fc:
            # -1 factor in flux.form means this is a lhs term
            form_type = "boundary_reaction" if flux.is_boundary_condition else "domain_reaction"
            flux_form_units = flux.equation_units * flux.measure_units
            # Determine if flux is linear w.r.t. compartment functions
            # Use flux.is_linear_wrt_comp and combine with linear_wrt_comp
            # (prioritizing former). If compartment is not relevant to flux then it is linear
            linearity_dict = {k: flux.is_linear_wrt_comp.setdefault(k, True) for k in self.cc.keys}
            # linearity_dict = nonlinear_wrt_comp#{k :
            # flux.is_linear_wrt_comp.setdefault(k, True) for k in self.cc.keys}
            self.forms.add(
                Form(
                    f"{flux.name}",
                    flux.form,
                    flux.destination_species,
                    form_type,
                    flux_form_units,
                    True,
                    linearity_dict,
                )
            )

        for species in self.sc:
            u = species._usplit["u"]
            # ut = species.ut
            # un = species.u['n']
            un = species._usplit["n"]
            v = species.v
            D = species.D
            dx = species.compartment.mesh.dx
            # diffusion term
            if species.D == 0:
                logger.debug(
                    f"Species {species.name} has a diffusion coefficient of 0. "
                    "Skipping creation of diffusive form.",
                    extra=dict(format_type="log"),
                )
            else:
                Dform = D * d.inner(d.grad(u), d.grad(v)) * dx
                # exponent is -2 because of two gradients
                Dform_units = (
                    species.diffusion_units
                    * species.concentration_units
                    * species.compartment.compartment_units
                    ** (species.compartment.dimensionality - 2)
                )
                self.forms.add(
                    Form(
                        f"diffusion_{species.name}",
                        Dform,
                        species,
                        "diffusion",
                        Dform_units,
                        True,
                        linear_wrt_comp,
                    )
                )
            # mass (time derivative) terms
            Muform = (u) * v / self.dT * dx
            mass_form_units = (
                species.concentration_units
                / unit.s
                * species.compartment.compartment_units**species.compartment.dimensionality
            )
            self.forms.add(
                Form(
                    f"mass_u_{species.name}",
                    Muform,
                    species,
                    "mass_u",
                    mass_form_units,
                    True,
                    linear_wrt_comp,
                )
            )
            Munform = (-un) * v / self.dT * dx
            self.forms.add(
                Form(
                    f"mass_un_{species.name}",
                    Munform,
                    species,
                    "mass_un",
                    mass_form_units,
                    True,
                    linear_wrt_comp,
                )
            )
        # FIXME: `has_diffusive_forms` is only in commented out code of
        # `initialize_discrete_variational_problem_and_solver`
        for compartment in self.cc:
            diffusive_forms = [
                f
                for f in self.forms
                if f.compartment.name == compartment.name and f.form_type == "diffusion"
            ]
            if len(diffusive_forms) == 0:
                logger.debug(
                    f"Compartment {compartment.name} has no diffusive forms.",
                    extra=dict(format_type="log"),
                )
                compartment.has_diffusive_forms = False
            else:
                compartment.has_diffusive_forms = True

    def initialize_discrete_variational_problem_and_solver(self):
        """
        Formulate problem for Newton iterations
        and configure solver - currently, only using SNES in PETSc
        is supported, the dolfin MixedNonlinearVariationalSolver
        is not tested in this context
        """
        logger.debug(
            "Formulating problem as F(u;v) == 0 for newton iterations",
            extra=dict(format_type="log"),
        )
        # self.all_forms = sum([f.form for f in self.forms])
        # self.problem = d.NonlinearVariationalProblem(self.all_forms, self.u['u'], bcs=None)
        # Aliases
        u = self.u["u"]._functions
        self.global_sizes = self.get_global_sizes(u)

        # Because it is a little tricky (see comment on d.extract_blocks(F)
        # in model.get_block_system()),
        # we are only going to separate fluxes that are linear with
        # respect to all compartments
        self.Fsum_all = sum([f.lhs for f in self.forms])  # Sum of all forms
        if self.config.solver["snes_preassemble_linear_system"]:
            # debug attempt
            logger.debug(
                "Getting linear+non-linear block Jacobian components",
                extra=dict(format_type="log"),
            )
            self.Fblocks_all, self.Jblocks_all, _ = self.get_block_system(self.Fsum_all, u)

        # Not separating linear/non-linear components (everything assumed non-linear)
        else:
            self.Fblocks_all = self.get_block_F(self.Fsum_all, u)
            self.Jblocks_all = self.get_block_J(self.Fsum_all, u)

        # Print the residuals per compartment
        for compartment in self._active_compartments:
            res = self.get_compartment_residual(compartment, norm=2)
            logger.debug(
                f"Initial L2-norm of compartment {compartment.name} is {res}",
                extra=dict(format_type="log"),
            )
            if res > 1:
                logger.warning(
                    f"Warning! Initial L2-norm of compartment {compartment.name} is "
                    f"{res} (possibly too large).",
                    extra=dict(format_type="log_urgent"),
                )

        # if use snes
        if self.config.solver["use_snes"]:
            logger.debug("Using SNES solver", extra=dict(format_type="log"))
            self.problem = smartSNESProblem(
                self.u["u"],
                self.Fblocks_all,
                self.Jblocks_all,
                self._active_compartments,
                self._all_compartments,
                self.stopwatches,
                self.config.solver["print_assembly"],
            )

            self.problem.init_petsc_matnest()
            self.problem.init_petsc_vecnest()
            if len(self.problem.global_sizes) == 1:
                self._ubackend = u[0].vector().vec().copy()
            else:
                self._ubackend = PETSc.Vec().createNest(
                    [usub.vector().vec().copy() for usub in u], comm=self.mpi_comm_world
                )

            self.solver = PETSc.SNES().create(self.mpi_comm_world)

            # Define the function/jacobian blocks
            self.solver.setFunction(self.problem.F, self.problem.Fpetsc_nest)
            self.solver.setJacobian(self.problem.J, self.problem.Jpetsc_nest)
            self.solver.setType("newtonls")
            self.solver.setTolerances(rtol=1e-5)

            def monitor(snes, it, fgnorm):
                # prints out residual at each Newton iteration
                logger.debug("  " + str(it) + " SNES Function norm " + "{:e}".format(fgnorm))

            self.solver.setMonitor(monitor)
            opts = PETSc.Options()
            opts["snes_linesearch_type"] = "l2"
            self.solver.setFromOptions()

            # May look into using LU in all cases if possible (seems to converge very slowly)
            if self.problem.is_single_domain:
                # If only modeling species within a single domain
                # (other domains may still contribute as BCs),
                # then just use hypre (cannot use field split without multiple domains)
                self.solver.ksp.setType("bcgs")
                self.solver.ksp.setTolerances(rtol=1e-5)
                self.solver.ksp.pc.setType("hypre")
            else:  # Field split preconditioning:
                # These are some reasonable preconditioner/linear solver settings for block systems
                # Krylov solver
                # biconjugate gradient stabilized. in most cases probably the best option
                self.solver.ksp.setType("bcgs")
                self.solver.ksp.setTolerances(rtol=1e-5)
                # Some other reasonable krylov solvers: (don't think they work with block systems)
                # bcgsl, ibcgs (improved stabilized bcgs)
                # fbcgsr, fbcgs (flexible bcgs)
                self.solver.ksp.pc.setType("fieldsplit")
                # Set the indices
                nest_indices = self.problem.Jpetsc_nest.getNestISs()[0]
                nest_indices_tuples = [(str(i), val) for i, val in enumerate(nest_indices)]
                # self.solver.ksp.pc.setFieldSplitIS(("0", is_0), ("1", is_1))
                self.solver.ksp.pc.setFieldSplitIS(*nest_indices_tuples)
                # 0 == 'additive' [jacobi], 1 == gauss-seidel
                self.solver.ksp.pc.setFieldSplitType(1)
                subksps = self.solver.ksp.pc.getFieldSplitSubKSP()
                for i, subksp in enumerate(subksps):
                    # subksp.setType('preonly')
                    # # If there is not diffusion then this is really just a distributed set of ODEs
                    # if not self._active_compartments[i].has_diffusive_forms:
                    #     subksp.pc.setType('none')
                    subksp.setType("preonly")
                    subksp.pc.setType("hypre")
        else:
            logger.debug(
                "Using dolfin MixedNonlinearVariationalSolver",
                extra=dict(format_type="log"),
            )
            self._ubackend = [u[i]._cpp_object for i in range(len(u))]
            self.problem = d.cpp.fem.MixedNonlinearVariationalProblem(
                self.Fblocks_all, self._ubackend, [], self.Jblocks_all
            )
            # self.problem_alternative = d.MixedNonlinearVariationalProblem(Fblock, u, [], J)
            self.solver = d.MixedNonlinearVariationalSolver(self.problem)

    # @staticmethod

    def get_block_system(self, Fsum, u):
        """
        Modify F and define J in specific structure required
        for  :code:`dolfin.assemble_mixed()` - use to preassemble linear system
        for the dolfin MixedNonlinearVariationalSolver (untested)


        The high level  :code:`dolfin.solve(F==0, u)` eventually
        calls cpp.fem.MixedNonlinearVariationalSolver,
        but first modifies F and defines J into a specific
        structure that is required for  :code:`dolfin.assemble_mixed()`

        Comments on :code:`d.extract_blocks(F)` (which is just a wrapper
        around ufl.algorithms.formsplitter)

        There is some indexing going on behind the scenes, so just
        manually summing what we know to be the
        components of Fblock[0] will not be the same as extract_blocks(F)[0].
        Here is an example:

        .. code:: python

            F  = sum([f.lhs for f in model.forms]) # single form
            # first compartment
            F0 = sum([f.lhs for f in model.forms if f.compartment.name=='cytosol'])
            Fb0 = extract_blocks(F)[0] # tuple of forms

            F0.equals(Fb0) -> False
            I0 = F0.integrals()[0].integrand()
            Ib0 = Fb0.integrals()[0].integrand()
            # (ufl.Indexed(Argument))) vs ufl.Indexed(ListTensor(ufl.Indexed(Argument)))
            I0.ufl_operands[0] == Ib0.ufl_operands[0] -> False
            I0.ufl_operands[1] == Ib0.ufl_operands[1] -> True
            I0.ufl_operands[0] == Ib0.ufl_operands[0](1) -> True
        """

        # blocks/partitions are by compartment, not species
        Fblock = d.extract_blocks(Fsum)

        # =====================================================================
        # doflin.fem.problem.MixedNonlinearVariationalProblem()
        # =====================================================================
        # basically is a wrapper around the cpp class that finalizes preparing
        # F and J into the right format
        # TODO: add dirichlet BCs

        # Add in placeholders for empty blocks of F
        if len(Fblock) != len(u):
            Ftemp = [None for i in range(len(u))]
            for Fi in Fblock:
                Ftemp[Fi.arguments()[0].part()] = Fi
            Fblock = Ftemp

        # debug attempt
        J = []
        for Fi in Fblock:
            for uj in u:
                if Fi is None:
                    # pass
                    J.append(None)
                else:
                    dFdu = expand_derivatives(d.derivative(Fi, uj))
                    J.append(dFdu)

        # Check number of blocks in the residual and solution are coherent
        assert len(J) == len(u) * len(u)
        assert len(Fblock) == len(u)

        # Decompose F blocks into subforms based on domain of integration
        # Fblock = [F0, F1, ... , Fn] where the index is the compartment index
        # Flist  = [[F0(Omega_0), F0(Omega_1)], ..., [Fn(Omega_n)]]
        # If a form has integrals on multiple domains, they are split into a list
        Flist = list()
        for idx, Fi in enumerate(Fblock):
            if Fi is None or Fi.empty():
                logger.warning(
                    f"F{idx} = F[{self.cc.get_index(idx).name}]) is empty",
                    extra=dict(format_type="warning"),
                )
                Flist.append([d.cpp.fem.Form(1, 0)])
            else:
                Fs = []
                for Fsub in sub_forms_by_domain(Fi):
                    if Fsub is None or Fsub.empty():
                        domain = self.get_mesh_by_id(Fsub.mesh().id()).name
                        logger.warning(
                            f"F{idx} = F[{self.cc.get_index(idx).name}] "
                            "is empty on integration domain {domain}",
                            extra=dict(format_type="logred"),
                        )
                        Fs.append(d.cpp.fem.Form(1, 0))
                    else:
                        Fs.append(d.Form(Fsub))
                Flist.append(Fs)

        # Decompose J blocks into subforms based on domain of integration
        Jlist = list()
        for idx, Ji in enumerate(J):
            idx_i, idx_j = divmod(idx, len(u))
            if Ji is None or Ji.empty():
                logger.warning(
                    f"J{idx_i}{idx_j} = dF[{self.cc.get_index(idx_i).name}])"
                    f"/du[{self.cc.get_index(idx_j).name}] is empty",
                    extra=dict(format_type="logred"),
                )
                Jlist.append([d.cpp.fem.Form(2, 0)])
            else:
                Js = []
                for Jsub in sub_forms_by_domain(Ji):
                    if Jsub is None or Jsub.empty():
                        domain = self.get_mesh_by_id(Jsub.mesh().id()).name
                        logger.warning(
                            f"J{idx_i}{idx_j} = dF[{self.cc.get_index(idx_i).name}])"
                            f"/du[{self.cc.get_index(idx_j).name}]"
                            f"is empty on integration domain {domain}",
                            extra=dict(format_type="logred"),
                        )
                    Js.append(d.Form(Jsub))
                Jlist.append(Js)

        global_sizes = [uj.function_space().dim() for uj in u]

        # return Flist, Jlist
        return Flist, Jlist, global_sizes

    def get_global_sizes(self, u):
        """Return total number of dof for current model"""
        return [uj.function_space().dim() for uj in u]

    def get_block_F(self, Fsum, u):
        """Assemble block F-vector by compartment
        (F is the residual)
        """
        # blocks/partitions are by compartment, not species
        Fblock = d.extract_blocks(Fsum)

        # Add in placeholders for empty blocks of F
        if len(Fblock) != len(u):
            Ftemp = [None for i in range(len(u))]
            for Fi in Fblock:
                Ftemp[Fi.arguments()[0].part()] = Fi
            Fblock = Ftemp

        assert len(Fblock) == len(u)
        # Decompose F blocks into subforms based on domain of integration
        # Fblock = [F0, F1, ... , Fn] where the index is the compartment index
        # Flist  = [[F0(Omega_0), F0(Omega_1)], ..., [Fn(Omega_n)]]
        # If a form has integrals on multiple domains, they are split into a list
        Flist = list()
        for idx, Fi in enumerate(Fblock):
            if Fi is None or Fi.empty():
                logger.warning(
                    f"F{idx} = F[{self.cc.get_index(idx).name}]) is empty",
                    extra=dict(format_type="warning"),
                )
                Flist.append([d.cpp.fem.Form(1, 0)])
            else:
                Fs = []
                for Fsub in sub_forms_by_domain(Fi):
                    if Fsub is None or Fsub.empty():
                        domain = self.get_mesh_by_id(Fsub.mesh().id()).name
                        logger.warning(
                            f"F{idx} = F[{self.cc.get_index(idx).name}] is empty "
                            f"on integration domain {domain}",
                            extra=dict(format_type="logred"),
                        )
                        Fs.append(d.cpp.fem.Form(1, 0))
                    else:
                        Fs.append(d.Form(Fsub))
                Flist.append(Fs)

        return Flist

    def get_block_J(self, Fsum, u):
        """Assemble block J by compartment
        (J is the Jacobian)
        """
        # blocks/partitions are by compartment, not species
        Fblock = d.extract_blocks(Fsum)
        J = []
        for Fi in Fblock:
            for uj in u:
                dFdu = expand_derivatives(d.derivative(Fi, uj))
                J.append(dFdu)

        # Check number of blocks in the residual and solution are coherent
        assert len(J) == len(u) * len(u)

        # Decompose J blocks into subforms based on domain of integration
        Jlist = list()
        for idx, Ji in enumerate(J):
            idx_i, idx_j = divmod(idx, len(u))
            if Ji is None or Ji.empty():
                logger.info(
                    f"J{idx_i}{idx_j} = dF[{self.cc.get_index(idx_i).name}])"
                    f"/du[{self.cc.get_index(idx_j).name}] is empty",
                    extra=dict(format_type="logred"),
                )
                Jlist.append([d.cpp.fem.Form(2, 0)])
            else:
                Js = []
                for Jsub in sub_forms_by_domain(Ji):
                    if Jsub is None or Jsub.empty():
                        domain = self.get_mesh_by_id(Jsub.mesh().id()).name
                        logger.warning(
                            f"J{idx_i}{idx_j} = dF[{self.cc.get_index(idx_i).name}])"
                            f"/du[{self.cc.get_index(idx_j).name}]"
                            f"is empty on integration domain {domain}",
                            extra=dict(format_type="logred"),
                        )
                    Js.append(d.Form(Jsub))
                Jlist.append(Js)

        return Jlist

    def set_form_scaling(self, compartment_name, scaling=1.0, print_scaling=True):
        for form in self.forms:
            if form.compartment.name == compartment_name:
                form.set_scaling(scaling, print_scaling)

    # ===============================================================================
    # Model - Solving (time related functions)
    # ===============================================================================
    def set_time(self, t):
        """Explicitly change time"""
        if t != self.t:
            logger.debug(f"Time changed from {self.t} to {t}", extra=dict(format_type="log"))
            self.t = t
            self.T.assign(t)

    def set_dt(self, dt):
        """Explicitly change time-step"""
        dt = self.rounded_decimal(dt)
        if self.config.solver["time_precision"] is not None:
            dt = round(dt, self.config.solver["time_precision"])

        if dt != self.dt:
            logger.debug(f"dt set to {dt} (previously {self.dt})", extra=dict(format_type="log"))
            self.dt = dt
            self.dT.assign(dt)

    def adjust_dt_if_prescribed(self):
        """Checks to see if the size of a full-time step would pass a "reset dt"
        checkpoint. At these checkpoints dt is reset to some value
        (e.g. to force smaller sampling during fast events)
        """
        # check that there are reset times specified
        if self.config.solver["adjust_dt"] is None or len(self.config.solver["adjust_dt"]) == 0:
            return
        # Aliases
        # check if we pass a reset dt checkpoint
        tnow = self.rounded_decimal(self.t)  # time right now
        dtnow = self.rounded_decimal(self.dt)
        tnext = tnow + dtnow  # the final time if dt is not reset
        tnext = self.rounded_decimal(tnext)
        # next time to adjust dt, and the value of dt to adjust to
        tadjust, dtadjust = self.config.solver["adjust_dt"][0]
        tadjust = self.rounded_decimal(tadjust)
        dtadjust = self.rounded_decimal(dtadjust)

        # if last time-step we reached a reset dt checkpoint then reset it now
        if self.reset_dt or tadjust == tnow:
            self.set_dt(dtadjust)
            logger.debug(
                f"[{self.idx}, t={tnow}] Adjusted time-step "
                f"(dt = {dtnow} -> {self.dt}) to match config specified value",
                extra=dict(format_type="log"),
            )
            del self.config.solver["adjust_dt"][0]
            self.reset_dt = False
            return

        if tadjust < tnow:
            raise AssertionError("tadjust (Next time to adjust dt) is smaller than current time.")

        # If true, this means taking a full time-step with current values of
        # t and dt would pass a checkpoint to adjust dt
        if tnow < tadjust <= tnext:
            # Safeguard against taking ridiculously small time-steps which may
            # cause convergence issues
            # (e.g. current time is tnow=0.999999999, tadjust=1.0,
            # dtadjust=0.01, instead of changing current dt to tadjust-tnow,
            # we change it to dtadjust)
            logger.debug(f"tadjust = {tadjust}")
            logger.debug(f"tnow = {tnow}")
            logger.debug(f"dtadjust = {dtadjust}")
            # this is needed otherwise very small time-steps might be
            # taken which wont converge
            new_dt = self.rounded_decimal(max([tadjust - tnow, dtadjust]))
            logger.debug(f"newdt = {new_dt}")

            if dtadjust > tadjust - tnow:
                logger.info(
                    f"[{self.idx}, t={tnow}] Adjusted time-step "
                    f"(dt = {dtnow} -> {new_dt}) to match config specified "
                    "value (adjusted early because dt_adjust > t_adjust-t_now)",
                    extra=dict(format_type="log"),
                )
                self.set_dt(new_dt)
                del self.config.solver["adjust_dt"][0]
                self.reset_dt = False
            else:
                logger.info(
                    f"[{self.idx}, t={tnow}] Adjusting time-step "
                    f"(dt = {dtnow} -> {new_dt}) to avoid passing reset dt checkpoint",
                    extra=dict(format_type="log_important"),
                )
                self.set_dt(new_dt)
                # set a flag to change dt to the config specified value
                self.reset_dt = True

    def adjust_dt_if_pass_tfinal(self):
        """Check if current value of t and dt would cause t+dt > t_final"""
        tnext = self.t + self.dt
        if tnext > self.final_t:
            new_dt = self.final_t - self.t
            if self.config.solver["time_precision"] is not None:
                new_dt = round(new_dt, self.config.solver["time_precision"])
            logger.info(
                f"[{self.idx}, t={self.t}] Adjusting time-step "
                f"(dt = {self.dt} -> {new_dt}) to avoid passing final time",
                extra=dict(format_type="log"),
            )
            self.set_dt(new_dt)

    def forward_time_step(self):
        """Take a step forward in time"""
        self.dt = self.rounded_decimal(self.dt)
        if self.config.solver["time_precision"] is not None:
            self.dt = round(self.dt, self.config.solver["time_precision"])
        self.tn = self.rounded_decimal(self.t)  # save the previous time
        self.t = self.rounded_decimal(self.t + self.dt)
        self.dT.assign(self.dt)
        self.T.assign(self.t)

        self.tvec.append(self.t)
        self.dtvec.append(self.dt)

        # self.update_time_dependent_parameters()

    def monolithic_solve(self):
        """Solve entire monolithic system using Newton iterations,
        with the specified PETSc SNES solver
        (or using the dolfin MixedNonlinearVariationalSolver, untested).
        If the solver diverges or the solution is negative,
        this function may call itself to solve the system with
        a smaller time step if

        .. code:: python

            self.config.solver["attempt_timestep_restart_on_divergence"]

        and/or

        .. code:: python

            self.config.solver["reset_timestep_for_negative_solution"]

        are true.
        """
        self.idx += 1
        # start a timer for the total time step
        self.stopwatches["Total time step"].start()
        # Adjust dt if necessary (if the last time-step did not
        # converge then it is already adjusted)
        if not self._failed_to_converge:
            # check if there is a manually prescribed time-step size
            self.adjust_dt_if_prescribed()
            self.adjust_dt_if_pass_tfinal()  # adjust dt so that it doesn't pass tfinal
        if self.dt <= 0:
            raise ValueError("dt is <= 0")

        # Take a step forward in time and update time-dependent parameters
        # update time-dependent parameters

        # march forward in time and update time-dependent parameters
        self.forward_time_step()
        logger.info(
            f"Beginning time-step {self.idx} [time={self.t}, dt={self.dt}]",
            extra=dict(
                format_type="timestep",
                new_lines=[1, 0],
            ),
        )
        self.update_time_dependent_parameters()
        if self.config.solver["use_snes"]:
            logger.info("Solving using PETSc.SNES Solver", dict(format_type="log"))
            self.stopwatches["snes all"].start()

            # Solve
            self.solver.solve(None, self._ubackend)

            # Store/compute timings
            logger.info(
                f"Completed time-step {self.idx} [time={self.t}, dt={self.dt}]",
                extra=dict(
                    new_lines=[1, 0],
                    format_type="solverstep",
                ),
            )
            for k in [
                "snes initialize zero matrices",
                "snes jacobian assemble",
                "snes residual assemble",
                "snes all",
            ]:
                print_results = False if k == "snes all" else True
                self.stopwatches[k].stop(print_results)

            assembly_time = (
                self.stopwatches["snes jacobian assemble"].stop_timings[-1]
                + self.stopwatches["snes residual assemble"].stop_timings[-1]
                + self.stopwatches["snes initialize zero matrices"].stop_timings[-1]
            )
            # time to solve minus all the assemblies
            solve_time = self.stopwatches["snes all"].stop_timings[-1] - assembly_time
            self.stopwatches["snes total assemble"].set_timing(assembly_time)
            self.stopwatches["snes total solve"].set_timing(solve_time)
            self.stopwatches["snes all"].print_last_stop()
            # self.stopwatch("SNES solver", stop=True)

            # Check how solver did
            self.idx_nl.append(self.solver.its)
            self.idx_l.append(self.solver.ksp.its)
            logger.info(
                f"Non-linear solver iterations: {self.solver.its}",
                extra=dict(format_type="log"),
            )
            logger.info(
                f"Linear solver iterations: {self.solver.ksp.its}",
                extra=dict(format_type="log"),
            )
            logger.info(
                f"SNES converged reason: {self.solver.getConvergedReason()}",
                extra=dict(format_type="log"),
            )
            logger.info(
                f"KSP converged reason: {self.solver.ksp.getConvergedReason()}",
                extra=dict(format_type="log"),
            )
            logger.info(
                f"KSP residual norm: {self.solver.ksp.getResidualNorm()}",
                extra=dict(format_type="log"),
            )

            # confirm that the solution is greater than or equal to zero,
            # otherwise reduce timestep and recompute
            if self.config.solver["reset_timestep_for_negative_solution"]:
                negVals = False
                for idx in range(self.num_active_compartments):
                    curSub = self.u["u"].sub(idx)
                    curVec = curSub.vector()
                    if any(
                        curVec < -1e-6
                    ):  # if value is "too negative", we reduce time step and recompute
                        negVals = True
                        break
                if negVals:
                    self.reset_timestep()
                    # Re-initialize SNES solver
                    if len(self.problem.global_sizes) == 1:
                        self._ubackend = self.u["u"]._functions[0].vector().vec().copy()
                    else:
                        self._ubackend = PETSc.Vec().createNest(
                            [usub.vector().vec().copy() for usub in self.u["u"]._functions],
                            comm=self.mpi_comm_world,
                        )
                    # need to re-link global function with species-specific functions
                    # after re-setting previous solution
                    self._init_4_4_get_species_u_v_V_dofmaps()
                    self._init_4_5_name_functions()
                    self._failed_to_converge = True
                    self.monolithic_solve()
                    return
                else:  # if value is >= -tol, then set any slightly negative solutions to zero
                    for idx in range(self.num_active_compartments):
                        curSub = self.u["u"].sub(idx)
                        curVec = curSub.vector()
                        for zeroIdx in np.asarray(curVec < 0).nonzero():
                            curSub.vector()[zeroIdx] = 0
                            d.assign(self.u["u"].sub(idx), curSub)
                    for idx in np.asarray(self._ubackend.array < 0).nonzero():
                        self._ubackend.array[idx] = 0

            if not self.solver.converged:
                if not self.config.solver["attempt_timestep_restart_on_divergence"]:
                    raise RuntimeError(
                        f"Model {self.name}: SNES diverged (Reason = "
                        f"{self.solver.getConvergedReason()}) and "
                        "attempt_timestep_restart_on_divergence is False. Exiting."
                    )
                self.stopwatches["Total time step"].stop()
                logger.info(
                    f"SNES failed to converge. Reason = {self.solver.getConvergedReason()}",
                    extra=dict(format_type="log"),
                )
                logger.debug(
                    "(https://petsc.org/main/docs/manualpages/SNES/SNESConvergedReason.html)",
                    extra=dict(format_type="log"),
                )
                self.reset_timestep()
                # Re-initialize SNES solver
                if len(self.problem.global_sizes) == 1:
                    self._ubackend = self.u["u"]._functions[0].vector().vec().copy()
                else:
                    self._ubackend = PETSc.Vec().createNest(
                        [usub.vector().vec().copy() for usub in self.u["u"]._functions]
                    )
                # need to re-link global function with species-specific functions
                # after re-setting previous solution
                self._init_4_4_get_species_u_v_V_dofmaps()
                self._init_4_5_name_functions()
                self._failed_to_converge = True
                self.monolithic_solve()
                return
            else:
                self._failed_to_converge = False
        else:
            logger.info(
                "Solving using dolfin.MixedNonlinearVariationalSolver()",
                extra=dict(format_type="log"),
            )
            self.solver.solve()

            logger.info(
                f"Total residual: {self.get_total_residual(norm=2)}",
                extra=dict(format_type="log"),
            )
            residuals = dict()
            for compartment in self._active_compartments:
                residuals[compartment.name] = self.get_compartment_residual(compartment, norm=2)
                logger.debug(
                    f"L2-norm of compartment {compartment.name} is {residuals[compartment.name]}",
                    extra=dict(format_type="log"),
                )
                if residuals[compartment.name] > 1:
                    logger.warning(
                        f"Warning! L2-norm of compartment {compartment.name} "
                        f"is {residuals[compartment.name]} (possibly too large).",
                        extra=dict(format_type="log_urgent"),
                    )

            self.residuals.append(residuals)

        # self.data['nl_idx'].append(nl_idx)
        # self.data['success'].append(success)
        # self.data['tvec'].update(self.t)
        # self.data['dtvec'].update(self.dt)

        self.update_solution()  # assign most recently computed solution to "previous solution"
        # self.adjust_dt() # adjusts dt based on number of nonlinear iterations required
        # self.stopwatch("Total time step", stop=True)
        self.stopwatches["Total time step"].stop()

    def reset_timestep(self, dt_scale=0.20):
        """t failed. Revert t->tn and revert solution"""
        logger.debug(f"Resetting time-step: {self.idx}", extra=dict(format_type="log"))
        # Change t and decrease dt
        self.set_time(self.tvec[-2])  # t=tn
        self.set_dt(float(self.dtvec[-1]) * dt_scale)  # dt=dt*0.2
        # Store information on failed solve
        # idx, idx_nl, idx_l, t, dt, residuals, reason
        self.failed_solves.append(
            (
                self.idx,
                self.idx_nl[-1],
                self.idx_l[-1],
                self.tvec[-1],
                self.dtvec[-1],
                # self.residuals[-1],
                self.solver.getConvergedReason(),
            )
        )
        # Remove previous values
        self.idx = int(self.idx) - 1
        for data in [
            self.idx_nl,
            self.idx_l,
            self.tvec,
            self.dtvec,
        ]:  # , self.residuals]:
            data.pop()
        # Undo the solution to the previous time-step
        self.update_solution(ukeys=["u"], unew="n")

    def update_time_dependent_parameters(self):
        r"""
        Updates all time dependent parameters. Time-dependent parameters are
        either defined either symbolically or through a data file, and each of
        these can either be defined as a direct function of :math:`t, p(t)`, or a
        "pre-integrated expression", :math:`\int_{t_n}^{t_{n+1}} P(\tau) d\tau`, which allows for
        exact integration when the expression the parameter appears in doesn't rely
        on any other time-dependet variables. This may be useful for guaranteeing
        a certain amount of flux independent of time-stepping.

        Backward Euler is essentially making the approximation:

        .. math::

            du/dt = f(u,t)  ->  (u(t_{n+1}) - u(t_n)) = \int_{t_n}^{t_{n+1}}
            f(u(t_{n+1}),t_{n+1}) dt \approx dt*f(u(t_{n+1}),t_{n+1})

        If some portion of f is only dependent on t, e.g. :math:`f=f_1+f_2+...+f_n, f_i=f_i(t)`,
        we can use the exact expression where :math:`F_i(t)` is
        the anti-derivative of :math:`f_i(t)`.

        .. math::

            \int_{t_n}^{t_{n+1}} f_i(t_{n+1}) -> (F_i(t_{n+1}) - F_i(t_n))

        Therefore,

        .. math::

            f(t_{n+1}) = (F_i(t_{n+1}) - F_i(t_n))/dt
        """
        # Aliases
        t = float(self.t)
        dt = float(self.dt)
        tn = float(self.tn)

        # Update time dependent parameters
        for parameter_name, parameter in self.pc.items:
            new_value = None
            if not parameter.is_time_dependent:
                continue
            if not parameter.use_preintegration:
                # Parameters that are defined as dolfin expressions will
                # automatically be updated by model.T.assign(t)
                if parameter.type == ParameterType.expression:
                    parameter.value = parameter.sym_expr.subs({"t": t}).evalf()
                    parameter.value_vector = np.vstack(
                        (parameter.value_vector, [t, parameter.value])
                    )
                    continue
                # Parameters from a data file need to have their dolfin constant updated
                if parameter.type == ParameterType.from_file:
                    t_data = parameter.sampling_data[:, 0]
                    p_data = parameter.sampling_data[:, 1]
                    # We never want time to extrapolate beyond the provided data.
                    if t < t_data[0] or t > t_data[-1]:
                        raise Exception("Parameter cannot be extrapolated beyond provided data.")
                    # Just in case... return a nan if value is outside of bounds
                    new_value = float(np.interp(t, t_data, p_data, left=np.nan, right=np.nan))
                    logger.debug(
                        f"Time-dependent parameter {parameter_name} updated by data. "
                        "New value is {new_value}",
                        extra=dict(format_type="log"),
                    )

            if parameter.use_preintegration:
                if parameter.type == ParameterType.expression:
                    a = parameter.preint_sym_expr.subs({"t": tn}).evalf()
                    b = parameter.preint_sym_expr.subs({"t": t}).evalf()
                    new_value = float((b - a) / dt)
                    logger.debug(
                        f"Time-dependent parameter {parameter_name} updated by "
                        "pre-integrated expression. New value is {new_value}",
                        extra=dict(format_type="log"),
                    )
                if parameter.type == ParameterType.from_file:
                    int_data = parameter.preint_sampling_data
                    a = np.interp(tn, int_data[:, 0], int_data[:, 1], left=np.nan, right=np.nan)
                    b = np.interp(t, int_data[:, 0], int_data[:, 1], left=np.nan, right=np.nan)
                    new_value = float((b - a) / dt)
                    logger.debug(
                        f"Time-dependent parameter {parameter_name} updated by "
                        f"pre-integrated data. New value is {new_value}",
                        extra=dict(format_type="log"),
                    )

            if new_value is not None:
                assert not np.isnan(new_value)
                parameter.value_vector = np.vstack((parameter.value_vector, [t, new_value]))
                parameter.value = new_value
                parameter.dolfin_constant.assign(new_value)
            else:
                # Time dependent but nothing assigned
                raise AssertionError()

    def update_solution(self, ukeys=["n"], unew="u"):
        """After finishing a time step, assign all the most recently computed solutions as
        the solutions for the previous time step.
        """
        if ukeys is None:
            ukeys = set(self.u.keys()).remove(unew)

        for ukey in ukeys:
            if ukey not in self.u.keys():
                raise ValueError(f"Key {ukey} is not in model.u.keys()")
            # for a function from a mixed function space
            for idx in range(self.num_active_compartments):
                self.u[ukey].sub(idx).assign(self.u[unew].sub(idx))

    # =========================================================
    # Model - Data manipulation
    # =========================================================
    # get the values of function u from subspace idx of some
    # mixed function space, V
    @staticmethod
    def dolfin_get_dof_indices(species):  # V, species_idx):
        """
        Returns indices *local* to the CPU (not global).
        Function values can then be returned e.g.

        .. code:: python

            indices = dolfin_get_dof_indices(V,species_idx)
            u.vector().get_local()[indices]
        """
        if species.dof_map is not None:
            return species.dof_map
        species_idx = species.dof_index

        V = sub(species.compartment.V, species_idx, collapse_function_space=False)

        indices = np.array(V.dofmap().dofs())
        # indices that this CPU owns
        first_idx, _ = V.dofmap().ownership_range()

        return indices - first_idx  # subtract index offset to go from global -> local indices

    def dolfin_set_function_values(self, sp, ukey, unew):
        """Set values for dolfin function (usually for initial condition)
        Input unew should either be an expression giving the spatial dependence
        of u, a constant (float), or a vector of values.
        :code:`d.assign(uold, unew)` works when uold is a subfunction
        :code:`uold.assign(unew)` does not (it will replace the entire function)
        """
        if isinstance(unew, d.Expression):
            uinterp = d.interpolate(unew, sp.V)
            d.assign(sp.u[ukey], uinterp)
        elif isinstance(unew, (float, int)):
            uinterp = d.interpolate(d.Constant(unew), sp.V)
            d.assign(sp.u[ukey], uinterp)
        elif len(sp.dof_map) == len(unew):
            # unew must be an N x 4 array: [X, Y, Z, function_values]
            u = self.cc[sp.compartment_name].u[ukey]
            dof_coord = sp.V.tabulate_dof_coordinates()  # coordinates for the current dof
            mesh_coord = unew[:, 0:3]  # x,y,z coordinates for initial condition
            function_values = unew[:, 3]
            # nearest neighbor interpolation
            # (in this case, matching up exactly with mesh points)
            dof_vals_cur = np.zeros((len(mesh_coord),))
            for i in range(len(dof_coord)):
                dist_vec = np.sum((dof_coord[i, :] - mesh_coord) ** 2, axis=1)
                dof_vals_cur[i] = function_values[np.argmin(dist_vec)]
                if i % 1000 == 0:
                    logger.debug(f"Set {i} of {len(dof_coord)} function values for {sp.name}")
            # indices = self.dolfin_get_dof_indices(sp)
            indices = sp.dof_map
            uvec = u.vector()
            values = uvec.get_local()
            values[indices] = dof_vals_cur
            uvec.set_local(values)
            uvec.apply("insert")
        elif isinstance(unew, d.Function):
            logger.debug(f"Function already set for {sp.name}")
        else:
            raise NotImplementedError

    @property
    def num_active_compartments(self):
        """number of compartments with dofs"""
        return len(self._active_compartments)

    def get_compartment_residual(self, compartment, norm=None):
        """returns compartment residual as given norm"""
        res_vec = sum(
            [d.assemble_mixed(form).get_local() for form in self.Fblocks_all[compartment.dof_index]]
        )
        if norm is None:
            return res_vec
        else:
            return np.linalg.norm(res_vec, norm)

    def get_total_residual(self, norm=None):
        """returns total residual for all active compartment as given norm"""
        res_vec = np.hstack(
            [d.assemble_mixed(form).get_local() for form in chain.from_iterable(self.Fblocks_all)]
        )
        res_vec = np.hstack(
            [
                self.get_compartment_residual(compartment, norm=None)
                for compartment in self._active_compartments
            ]
        )
        assert len(res_vec.shape) == 1
        if norm is None:
            return res_vec
        else:
            return np.linalg.norm(res_vec, norm)

    def get_mesh_by_id(self, mesh_id):
        """returns mesh with id, mesh_id"""
        for mesh in self.parent_mesh.all_meshes.values():
            if mesh.id == mesh_id:
                return mesh
            else:
                raise ValueError(f"No mesh with id {mesh_id}")

    # ============================================================
    # Model - Printing
    # ============================================================
    def print_meshes(self, tablefmt="fancy_grid"):
        """Print information associated with child meshes:
        name, id, dimensionality,
        num_cells, num_facets, and num_vertices
        """
        if self.mpi_am_i_root:
            properties_to_print = [
                "name",
                "id",
                "dimensionality",
                "num_cells",
                "num_facets",
                "num_vertices",
            ]
            df = pandas.DataFrame()

            # parent mesh
            tempdict = odict()
            for key in properties_to_print:
                tempdict[key] = getattr(self.parent_mesh, key)

            tempseries = pandas.Series(tempdict, name=self.parent_mesh.name)
            df = pandas.concat([df, tempseries.to_frame().T], ignore_index=True)
            # child meshes
            for child_mesh in self.child_meshes.values():
                tempdict = odict()
                for key in properties_to_print:
                    tempdict[key] = getattr(child_mesh, key)
                tempseries = pandas.Series(tempdict, name=child_mesh.name)
                df = pandas.concat([df, tempseries.to_frame().T], ignore_index=True)
            # intersection meshes
            for child_mesh in self.parent_mesh.child_surface_meshes:
                for mesh_id_pair in child_mesh.intersection_map.keys():
                    tempdict = odict()
                    if len(mesh_id_pair) == 1:
                        mesh_str = self.parent_mesh.get_mesh_from_id(list(mesh_id_pair)[0]).name
                    else:
                        mesh_str = (
                            self.parent_mesh.get_mesh_from_id(list(mesh_id_pair)[0]).name
                            + "_"
                            + self.parent_mesh.get_mesh_from_id(list(mesh_id_pair)[1]).name
                        )

                    intersection_mesh_name = f"{child_mesh.name}_intersect_{mesh_str}"
                    tempdict["name"] = intersection_mesh_name
                    tempdict["id"] = int(child_mesh.intersection_submesh[mesh_id_pair].id())
                    tempdict["dimensionality"] = child_mesh.dimensionality
                    tempdict["num_cells"] = child_mesh.intersection_submesh[
                        mesh_id_pair
                    ].num_cells()
                    tempdict["num_facets"] = child_mesh.intersection_submesh[
                        mesh_id_pair
                    ].num_facets()
                    tempdict["num_vertices"] = child_mesh.intersection_submesh[
                        mesh_id_pair
                    ].num_vertices()
                    tempseries = pandas.Series(tempdict, name=intersection_mesh_name)
                    df = pandas.concat([df, tempseries.to_frame().T], ignore_index=True)

            print(tabulate(df, headers="keys", tablefmt=tablefmt))

    def rounded_decimal(self, x):
        """Round time value to specified decimal point"""
        return Decimal(x).quantize(self._base_t)
