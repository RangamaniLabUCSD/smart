"""
Model class. Consists of parameters, species, etc. and is used for simulation
"""
from collections import defaultdict as ddict
from dataclasses import dataclass
from cached_property import cached_property
import itertools

import dolfin as d
import petsc4py.PETSc as PETSc

Print = PETSc.Sys.Print
import operator

import numpy as np
import sympy
from sympy.parsing.sympy_parser import parse_expr
from scipy.integrate import solve_ivp
from termcolor import colored
import pint

import stubs
import stubs.common as common
import stubs.model_assembly
from stubs.mesh import ChildMesh
from stubs import unit

color_print = common.color_print
from stubs.common import _fancy_print as fancy_print
from stubs.common import sub

comm = d.MPI.comm_world
rank = comm.rank
size = comm.size
root = 0


@dataclass
class Model:
    """
    Main stubs class. Consists of parameters, species, compartments, reactions, and can be simulated.
    """
    pc: stubs.model_assembly.ParameterContainer
    sc: stubs.model_assembly.SpeciesContainer
    cc: stubs.model_assembly.CompartmentContainer
    rc: stubs.model_assembly.ReactionContainer
    config: stubs.config.Config
    solver_system: stubs.solvers.SolverSystem
    parent_mesh: stubs.mesh.ParentMesh

    def __post_init__(self):
        # Check that solver_system is valid
        self.solver_system.check_solver_system_validity()
        self.solver_system.make_dolfin_parameter_dict()

        self.params = ddict(list)

        # FunctionSpaces, Functions, etc
        self.V = dict()
        self.u = dict()

        self.fc = stubs.model_assembly.FluxContainer()

        # Solver related parameters
        self.idx = 0
        self.nl_idx = {} # dictionary: compartment name -> # of nonlinear iterations needed
        self.success = {} # dictionary: compartment name -> success or failure to converge nonlinear solver
        self.stopping_conditions = {'F_abs': {}, 'F_rel': {}, 'udiff_abs': {}, 'udiff_rel': {}}
        self.t = 0.0
        self.dt = self.solver_system.initial_dt
        self.T = d.Constant(self.t)
        self.dT = d.Constant(self.dt)
        self.final_t = self.solver_system.final_t
        self.linear_iterations = None
        self.reset_dt = False

        # Timers
        self.timers = {}
        self.timings = ddict(list)

        # Functional forms
        self.Forms = stubs.model_assembly.FormContainer()
        self.a = {}
        self.L = {}
        self.F = {}

        # Solvers
        self.nonlinear_solver = {}
        self.linear_solver = {}
        self.scipy_odes = {}

        # Post processed data
        self.data = stubs.data_manipulation.Data(self, self.config)
        
        # Set loggers to logging levels defined in config
        self.config.set_logger_levels()
 
    @property
    def child_meshes(self):
        return self.parent_mesh.child_meshes

    @cached_property
    def min_dim(self):
        dim                         = min([comp.dimensionality for comp in self.cc])
        self.parent_mesh.min_dim    = dim
        return dim
    @cached_property
    def max_dim(self):
        dim                         = max([comp.dimensionality for comp in self.cc])
        self.max_dim                = dim
        self.parent_mesh.max_dim    = dim
        return dim

    # ==============================================================================
    # Model - Initialization
    # ==============================================================================

    # def initialize(self):
    #     # parameter/unit assembly
    #     print("\n\n********** Model initialization (Part 1/6) **********")
    #     print("Assembling parameters and units...\n")
    #     self.pc.do_to_all('assemble_units', {'unit_name': 'unit'})
    #     self.pc.do_to_all('assemble_units', {'value_name':'value', 'unit_name':'unit', 'assembled_name': 'value_unit'})
    #     self.pc.do_to_all('assemble_time_dependent_parameters')
    #     self.sc.do_to_all('assemble_units', {'unit_name': 'concentration_units'})
    #     self.sc.do_to_all('assemble_units', {'unit_name': 'diffusion_units'})
    #     self.cc.do_to_all('assemble_units', {'unit_name':'compartment_units'})
    #     self.rc.do_to_all('initialize_flux_equations_for_known_reactions', {"reaction_database": self.config.reaction_database})

    #     # linking containers with one another
    #     print("\n\n********** Model initialization (Part 2/6) **********")
    #     print("Linking different containers with one another...\n")
    #     self.rc._link_object(self.pc,'param_map','name','parameters', value_is_key=True)
    #     self.sc._link_object(self.cc,'compartment_name','name','compartment')
    #     self.sc._copy_linked_property('compartment', 'dimensionality', 'dimensionality')
    #     self.rc.do_to_all('get_involved_species_and_compartments', {"sc": self.sc, "cc": self.cc})
    #     self.rc._link_object(self.sc,'involved_species','name','involved_species_link')

    #     # meshes
    #     print("\n\n********** Model initialization (Part 3/6) **********")
    #     print("Loading in mesh and computing statistics...\n")
    #     setattr(self.cc, 'meshes', {self.parent_mesh.name: self.parent_mesh.mesh})
    #     self.cc.extract_submeshes(save_to_file=False)
    #     self.cc.compute_scaling_factors()

    #     # Associate species and compartments
    #     print("\n\n********** Model initialization (Part 4/6) **********")
    #     print("Associating species with compartments...\n")
    #     num_species_per_compartment = self.rc.get_species_compartment_counts(self.sc, self.cc)
    #     self.cc.get_min_max_dim()
    #     self.sc.assemble_compartment_indices(self.rc, self.cc)
    #     self.cc.add_property_to_all('is_in_a_reaction', False)
    #     self.cc.add_property_to_all('V', None)

    #     # dolfin functions
    #     print("\n\n********** Model initialization (Part 5/6) **********")
    #     print("Creating dolfin functions and assinging initial conditions...\n")
    #     self.sc.assemble_dolfin_functions(self.rc, self.cc)
    #     self.u = self.sc.u
    #     self.v = self.sc.v
    #     self.V = self.sc.V
    #     self.assign_initial_conditions()

    #     print("\n\n********** Model initialization (Part 6/6) **********")
    #     print("Assembling reactive and diffusive fluxes...\n")
    #     self.rc.reaction_to_fluxes()
    #     #self.rc.do_to_all('reaction_to_fluxes')
    #     self.fc = self.rc.get_flux_container()
    #     self.fc.do_to_all('get_additional_flux_properties', {"cc": self.cc, "solver_system": self.solver_system})
    #     self.fc.do_to_all('flux_to_dolfin')
 
    #     self.set_allow_extrapolation()
    #     # Turn fluxes into fenics/dolfin expressions
    #     self.assemble_reactive_fluxes()
    #     self.assemble_diffusive_fluxes()
    #     self.sort_forms()

    #     self.init_solutions_and_plots()


    def initialize_refactor(self):
        """
        Notes:
        * Now works with sub-volumes
        * Removed scale_factor (too ambiguous)
        """
        self._init_1()
        self._init_2()
        self._init_3()
        self._init_4()

    def _init_1(self):
        "Checking validity of model"
        fancy_print(f"Checking validity of model (step 1 of ZZ)", format_type='title')
        self._init_1_1_check_mesh_dimensionality()
        self._init_1_2_check_namespace_conflicts()
        self._init_1_3_check_parameter_dimensionality()
        fancy_print(f"Step 1 of initialization completed successfully!", text_color='magenta')
    def _init_2(self):
        "Cross-container dependent initializations (requires information from multiple containers)"
        fancy_print(f"Cross-Container Dependent Initializations (step 2 of ZZ)", format_type='title')
        self._init_2_1_reactions_to_symbolic_strings()
        self._init_2_2_check_reaction_validity()
        self._init_2_3_link_reaction_properties()
        self._init_2_4_check_for_unused_parameters_species_compartments()
        self._init_2_5_link_compartments_to_species()
        self._init_2_6_link_species_to_compartments()
        self._init_2_7_get_species_compartment_indices()
        fancy_print(f"Step 2 of initialization completed successfully!", text_color='magenta')
    def _init_3(self):
        "Mesh-related initializations"
        fancy_print(f"Mesh-related Initializations (step 3 of ZZ)", format_type='title')
        self._init_3_1_define_child_meshes()
        self._init_3_2_get_parent_mesh_functions()
        self._init_3_3_extract_submeshes()
        self._init_3_4_get_child_mesh_functions()
        self._init_3_5_get_integration_measures()
        fancy_print(f"Step 3 of initialization completed successfully!", format_type='log_important')
    def _init_4(self):
        "Dolfin function initializations"
        fancy_print(f"Dolfin Initializations (step 4 of ZZ)", format_type='title')
        self._init_4_1_get_dof_ordering()
        self._init_4_2_define_dolfin_function_spaces()
        self._init_4_3_define_dolfin_functions()
        self._init_4_4_get_species_u_v_V_dofmaps()
        self._init_4_5_name_functions()
        self._init_4_6_check_dolfin_function_validity()
        self._init_4_7_set_initial_conditions()
        

    # Step 1 - Checking model validity
    def _init_1_1_check_mesh_dimensionality(self):
        fancy_print(f"Check that mesh/compartment dimensionalities match", format_type='log')
        if (self.max_dim - self.min_dim) not in [0,1]:
            raise ValueError("(Highest mesh dimension - smallest mesh dimension) must be either 0 or 1.")
        if self.max_dim > self.parent_mesh.dimensionality:
            raise ValueError("Maximum dimension of a compartment is higher than the topological dimension of parent mesh.")
        # This is possible to simulate but could be unintended. (e.g. if you have a 3d mesh you could choose to only simulate on the surface)
        if self.max_dim != self.parent_mesh.dimensionality:
            fancy_print(f"Parent mesh has geometric dimension: {self.parent_mesh.dimensionality} which"
                            +f" is not the same as the maximum compartment dimension: {self.max_dim}.", format_type='warning')

        for compartment in self.cc:
            compartment.is_volume_mesh = compartment.dimensionality == self.max_dim

    def _init_1_2_check_namespace_conflicts(self):
        fancy_print(f"Checking for namespace conflicts", format_type='log')
        self._all_keys = set()
        containers = [self.pc, self.sc, self.cc, self.rc]
        for keys in [c.keys for c in containers]:
            self._all_keys = self._all_keys.union(keys)
        if sum([c.size for c in containers]) != len(self._all_keys):
            raise ValueError("Model has a namespace conflict. There are two parameters/species/compartments/reactions with the same name.")
        
        # Protect the variable names 'x[0]', 'x[1]', 'x[2]' and 't' because they are used for spatial dimensions and time
        if not {'x[0]', 'x[1]', 'x[2]', 't'}.isdisjoint(self._all_keys):
            raise ValueError("An object is using a protected variable name ('x[0]', 'x[1]', 'x[2]', or 't'). Please change the name.")
        
        # Make sure there are no overlapping markers
        self._all_markers = set()
        for compartment in self.cc.values:
            if isinstance(compartment.cell_marker, list):
                marker_list = compartment.cell_marker
            else:
                marker_list = [compartment.cell_marker]

            for marker in marker_list:
                if marker in self._all_markers:
                    raise ValueError(f"Two compartments have the same marker: {marker}")
                else:
                    self._all_markers.add(marker)

    def _init_1_3_check_parameter_dimensionality(self):
        if 'x[2]' in self._all_keys and self.max_dim<3:
            raise ValueError("An object has the variable name 'x[2]' but there are less than 3 spatial dimensions.")
        if 'x[1]' in self._all_keys and self.max_dim<2:
            raise ValueError("An object has the variable name 'x[1]' but there are less than 2 spatial dimensions.")
        

    # Step 2 - Cross-container Dependent Initialization
    def _init_2_1_reactions_to_symbolic_strings(self):
        fancy_print(f"Turning reactions into unsigned symbolic flux strings", format_type='log')
        """
        Turn all reactions into unsigned symbolic flux strings
        """
        for reaction in self.rc:
            # Mass action has a forward and reverse flux
            if reaction.reaction_type == 'mass_action':
                reaction.eqn_f_str = reaction.param_map['on']
                for sp_name in reaction.lhs:
                    reaction.eqn_f_str += '*' + sp_name
                #rxn.eqn_f = parse_expr(rxn_sym_str)

                reaction.eqn_r_str = reaction.param_map['off']
                for sp_name in reaction.rhs:
                    reaction.eqn_r_str += '*' + sp_name
                #rxn.eqn_r = parse_expr(rxn_sym_str)

            elif reaction.reaction_type == 'mass_action_forward':
                reaction.eqn_f_str = reaction.param_map['on']
                for sp_name in reaction.lhs:
                    reaction.eqn_f_str += '*' + sp_name
                #reaction.eqn_f = parse_expr(rxn_sym_str)

            # Custom reaction
            elif reaction.reaction_type in self.config.reaction_database.keys():
                reaction.eqn_f_str = reaction._parse_custom_reaction(self.config.reaction_database[reaction.reaction_type])
            
            # pre-defined equation string
            elif reaction.eqn_f_str or reaction.eqn_r_str:
                if reaction.eqn_f_str:
                    reaction.eqn_f_str = reaction._parse_custom_reaction(reaction.eqn_f_str)
                if reaction.eqn_r_str:
                    reaction.eqn_r_str = reaction._parse_custom_reaction(reaction.eqn_r_str)

            else:
                raise ValueError("Reaction %s does not seem to have an associated equation" % reaction.name)

    def _init_2_2_check_reaction_validity(self):
        fancy_print(f"Make sure all reactions have parameters/species defined", format_type='log')
        # Make sure all reactions have parameters/species defined
        for reaction in self.rc:
            for eqn_str in [reaction.eqn_f_str, reaction.eqn_r_str]:
                if not eqn_str:
                    continue
                eqn         = parse_expr(eqn_str)
                var_set     = {str(x) for x in eqn.free_symbols}
                param_set   = var_set.intersection(self.pc.keys)
                species_set = var_set.intersection(self.sc.keys)
                if len(param_set) + len(species_set) != len(var_set):
                    diff_set = var_set.difference(param_set.union(species_set))
                    raise NameError(f"Reaction {reaction.name} refers to a parameter or species ({diff_set}) that is not in the model.")

    def _init_2_3_link_reaction_properties(self):
        fancy_print(f"Linking parameters, species, and compartments to reactions", format_type='log')

        for reaction in self.rc:
            reaction.parameters   = {param_name: self.pc[param_name] for param_name in reaction.param_map.values()}
            reaction.species      = {species_name: self.sc[species_name] for species_name in reaction.species_map.values()}
            compartment_names     = [species.compartment_name for species in reaction.species.values()]
            if reaction.explicit_restriction_to_domain:
                compartment_names.append(reaction.explicit_restriction_to_domain)
            reaction.compartments = {compartment_name: self.cc[compartment_name] for compartment_name in compartment_names}
            # number of parameters, species, and compartments
            reaction.num_parmeters    = len(reaction.parameters)
            reaction.num_species      = len(reaction.species)
            reaction.num_compartments = len(reaction.compartments)

            is_volume_mesh = [c.is_volume_mesh for c in reaction.compartments.values()]
            if len(is_volume_mesh) == 1:
                if is_volume_mesh[0]:
                    reaction.topology = 'volume'
                else:
                    reaction.topology = 'surface'
            elif len(is_volume_mesh) == 2:
                if all(is_volume_mesh):
                    reaction.topology = 'volume_volume'
                elif not any(is_volume_mesh):
                    raise Exception(f"Reaction {reaction.name} involves two surfaces. This is not supported.")
                else:
                    reaction.topology = 'volume_surface'
            elif len(is_volume_mesh) == 3:
                if sum(is_volume_mesh) == 3:
                    raise Exception(f"Reaction {reaction.name} involves three volumes. This is not supported.")
                elif sum(is_volume_mesh) < 2:
                    raise Exception(f"Reaction {reaction.name} involves two or more surfaces. This is not supported.")
                else:
                    reaction.topology = 'volume_surface_volume'
            else:
                raise ValueError("Number of compartments involved in a flux must be in [1,2,3]!")

                    
                
        
    def _init_2_4_check_for_unused_parameters_species_compartments(self):
        fancy_print(f"Checking for unused parameters, species, or compartments", format_type='log')

        all_parameters   = set(itertools.chain.from_iterable([r.parameters for r in self.rc]))
        all_species      = set(itertools.chain.from_iterable([r.species for r in self.rc]))
        all_compartments = set(itertools.chain.from_iterable([r.compartments for r in self.rc]))
        if all_parameters != set(self.pc.keys):
            raise ValueError(f"Parameter(s), {set(self.pc.keys).difference(all_parameters)}, are unused in any reactions.") 
        if all_species != set(self.sc.keys):
            raise ValueError(f"Species, {set(self.sc.keys).difference(all_species)}, are unused in any reactions.") 
        if all_compartments != set(self.cc.keys):
            raise ValueError(f"Compartment(s), {set(self.cc.keys).difference(all_compartments)}, are unused in any reactions.") 

    def _init_2_5_link_compartments_to_species(self):
        fancy_print(f"Linking compartments and compartment dimensionality to species", format_type='log')
        for species in self.sc:
            species.compartment = self.cc[species.compartment_name]
            species.dimensionality = self.cc[species.compartment_name].dimensionality

    def _init_2_6_link_species_to_compartments(self):
        fancy_print(f"Linking species to compartments", format_type='log')
        # An species is considered to be "in a compartment" if it is involved in a reaction there
        for species in self.sc:
            species.compartment.species.update({species.name: species})
        for compartment in self.cc:
            compartment.num_species = len(compartment.species)
    
    def _init_2_7_get_species_compartment_indices(self):
        fancy_print(f"Getting indices for species for each compartment", format_type='log')
        for compartment in self.cc:
            index=0
            for species in list(compartment.species.values()):
                species.dof_index = index
                index += 1

    # Step 3 - Mesh Initializations
    def _init_3_1_define_child_meshes(self):
        fancy_print(f"Defining child meshes", format_type='log')
        # Check that there is a parent mesh loaded
        if not isinstance(self.parent_mesh, stubs.mesh.ParentMesh):
            raise ValueError("There is no parent mesh.")

        # Define child meshes
        for comp in self.cc:
            comp.mesh = ChildMesh(self.parent_mesh, comp)

        # Check validity (one child mesh for each compartment)
        assert len(self.child_meshes) == self.cc.size

    def _init_3_2_get_parent_mesh_functions(self):
        fancy_print(f"Defining parent mesh functions", format_type='log')
        self.parent_mesh.get_mesh_functions()

    def _init_3_3_extract_submeshes(self):
        """ Use dolfin.MeshView.create() to extract submeshes """
        fancy_print(f"Extracting submeshes", format_type='log')
        # Aliases
        mesh = self.parent_mesh.dolfin_mesh
        # Loop through child meshes and extract submeshes
        for cm in self.child_meshes.values():
            cm.extract_submesh()

    def _init_3_4_get_child_mesh_functions(self):
        fancy_print(f"Defining child mesh functions", format_type='log')
        # this requires mapping information from the parent mesh functions
        for mesh in self.parent_mesh.child_meshes.values():
            mesh.get_mesh_functions()

    def _init_3_5_get_integration_measures(self):
        fancy_print(f"Getting integration measures for parent mesh and child meshes", format_type='log')
        for mesh in self.parent_mesh.all_meshes.values():
            mesh.get_integration_measures()

    def _init_4_1_get_dof_ordering(self):
        """
        Arrange the compartments based on the number of degrees of freedom they have
        (We want to have the highest number of dofs first)
        """
        self.sorted_compartments = self.cc.sort_by('num_dofs')[0]
        for idx, compartment in enumerate(self.sorted_compartments):
            compartment.dof_index = idx

    # Step 4 - Dolfin Functions 
    def _init_4_2_define_dolfin_function_spaces(self):
        fancy_print(f"Defining dolfin function spaces for compartments", format_type='log')
        # Aliases
        max_compartment_name = max([len(compartment_name) for compartment_name in self.cc.keys])
        
        # Make the individual function spaces (per compartment)
        for compartment in self.sorted_compartments:
            # Aliases
            fancy_print(f"Defining function space for {compartment.name}{' '*(max_compartment_name-len(compartment.name))} "
                        f"(dim: {compartment.dimensionality}, species: {compartment.num_species}, dofs: {compartment.num_dofs})", format_type='log')

            if compartment.num_species > 1:
                compartment.V = d.VectorFunctionSpace(compartment.dolfin_mesh, 'P', 1, dim=compartment.num_species)
            else:
                compartment.V = d.FunctionSpace(compartment.dolfin_mesh, 'P', 1)

        self.V = [compartment.V for compartment in self.sorted_compartments]
        # Make the MixedFunctionSpace
        self.W = d.MixedFunctionSpace(*self.V)

    def _init_4_3_define_dolfin_functions(self):
        fancy_print(f"Defining dolfin functions", format_type='log')
        # dolfin functions created from MixedFunctionSpace
        self.u['u'] = d.Function(self.W)
        self.u['k'] = d.Function(self.W)
        self.u['n'] = d.Function(self.W)

        # Trial and test functions
        self.ut     = d.TrialFunctions(self.W)
        self.v      = d.TestFunctions(self.W)

        # Create references in compartments to the subfunctions
        for compartment in self.sorted_compartments:
            # alias
            cidx = compartment.dof_index

            # functions
            for key, func in self.u.items():
                compartment.u[key] = sub(func,cidx) #func.sub(cidx)

            # trial and test functions
            compartment.ut = self.ut[cidx]
            compartment.v  = self.v[cidx]
            
    def _init_4_4_get_species_u_v_V_dofmaps(self):
        fancy_print(f"Extracting subfunctions/function spaces/dofmap for each species", format_type='log')
        for compartment in self.sorted_compartments:
            # loop through species and add the name/index
            for species in compartment.species.values():
                species.V = sub(compartment.V, species.dof_index)
                species.v = sub(compartment.v, species.dof_index)
                species.dof_map = self.dolfin_get_dof_indices(species) # species.V.dofmap().dofs()

                for key in compartment.u.keys():
                    species.u[key] = sub(compartment.u[key], species.dof_index) #compartment.u[key].sub(species.dof_index)
                    
    def _init_4_5_name_functions(self):
        fancy_print(f"Naming functions and subfunctions", format_type='log')
        for compartment in self.sorted_compartments:
            # name of the compartment function
            for key in self.u.keys():
                compartment.u[key].rename(f"{compartment.name}_{key}", "")
                # loop through species and add the name/index
                for species in compartment.species.values():
                    sidx = species.dof_index
                    if compartment.num_species > 1:
                        species.u[key].rename(f"{compartment.name}_{sidx}_{species.name}_{key}", "")

    def _init_4_6_check_dolfin_function_validity(self):
        "Sanity check... If an error occurs here it is likely an internal bug..."
        fancy_print(f"Checking that dolfin functions were created correctly", format_type='log')
        # sanity check
        for compartment in self.cc:
            idx = compartment.dof_index
            # function size == dofs
            assert compartment.u['u'].vector().size() == compartment.num_dofs

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
        "Sets the function values to initial conditions"
        fancy_print(f"Set function values to initial conditions", format_type='log')
        for species in self.sc:
            for ukey in species.u.keys():
                if isinstance(species.initial_condition, float):
                    self.dolfin_set_function_values(species, ukey, species.initial_condition)
                else:
                    self.dolfin_set_function_values(species, ukey, species.initial_condition_expression)

        # # project to boundary/volume functions
        # self.update_solution_volume_to_boundary()
        # self.update_solution_boundary_to_volume()

    # TODO
    #     self.rc.reaction_to_fluxes()
    #     #self.rc.do_to_all('reaction_to_fluxes')
    #     self.fc = self.rc.get_flux_container()
    #     self.fc.do_to_all('get_additional_flux_properties', {"cc": self.cc, "solver_system": self.solver_system})
    #     self.fc.do_to_all('flux_to_dolfin')
 
    #     self.set_allow_extrapolation()
    #     # Turn fluxes into fenics/dolfin expressions
    #     self.assemble_reactive_fluxes()
    #     self.assemble_diffusive_fluxes()
    #     self.sort_forms()

    #     self.init_solutions_and_plots()
    def _init_5_1_reactions_to_fluxes(self):
        fancy_print(f"Convert reactions to flux objects", format_type='log')
        for reaction in self.rc:
            reaction.reaction_to_fluxes()
            self.fc.add(reaction.fluxes)
            
    def _init_5_2_set_flux_units(self):
        fancy_print(f"Checking flux units for dimensional consistency.", format_type='log')
        for flux in self.fc:
            concentration_units = flux.destination_species.concentration_units
            compartment_units   = flux.destination_compartment.compartment_units
            diffusion_units     = flux.destination_species.diffusion_units

            if flux.is_boundary_condition:
                flux_units = concentration_units / compartment_units * diffusion_units
            else:
                flux_units = concentration_units / unit.s

            # correct units
            if flux_units.dimensionality != flux.equation_units.dimensionality:
                raise ValueError(f"Flux {flux.name} has wrong units "
                                 f"(expected {flux_units}, got {flux.equation_units}.")
            else:
                flux.unit_scale_factor = flux.equation_units.to(flux_units)/flux.equation_units
                assert flux.unit_scale_factor.dimensionless
                assert flux.equation_units*flux.unit_scale_factor == flux.equation_units.to(flux_units)

                if flux.unit_scale_factor.magnitude == 1.0:
                    continue

                fancy_print(f"\nFlux {flux.name} scaled by {flux.unit_scale_factor}", format_type='log')
                fancy_print(f"Old flux units: {flux.equation_units}", format_type='log')
                fancy_print(f"New flux units: {flux_units}", format_type='log')
                print("")

                # update flux.equation_value and flux.equation_units
                flux.equation_eval *= flux.unit_scale_factor
                flux.evaluate_equation()
                assert flux.equation_units == flux_units

    
    # def _init_5_3_reaction_fluxes_to_forms(self):
    #     for flux in self.fc:
            #CONTINUE

        
    def assemble_reactive_fluxes(self):
        """
        Creates the actual dolfin objects for each flux. Checks units for consistency
        """
        for j in self.fc:
            total_scaling = 1.0 # all adjustments needed to get congruent units
            sp = j.species_map[j.species_name]
            prod = j.prod
            unit_prod = j.unit_prod
            # first, check unit consistency
            if (unit_prod/j.flux_units).dimensionless:
                setattr(j, 'scale_factor', 1*unit.dimensionless)
                pass
            else:
                if hasattr(j, 'length_scale_factor'):
                    Print("Adjusting flux for %s by the provided length scale factor." % (j.name, j.length_scale_factor))
                    length_scale_factor = getattr(j, 'length_scale_factor')
                else:
                    if len(j.involved_compartments.keys()) < 2:
                        Print("Units of flux: %s" % unit_prod)
                        Print("Desired units: %s" % j.flux_units)
                        raise Exception("Flux %s seems to be a boundary flux (or has inconsistent units) but only has one compartment, %s."
                            % (j.name, j.destination_compartment))
                    length_scale_factor = j.involved_compartments[j.source_compartment].scale_to[j.destination_compartment]

                Print(f'\nThe flux, {j.flux_name}, from compartment {j.source_compartment} to {j.destination_compartment}, has units {colored(unit_prod.to_root_units(), "red")}... the desired units for this flux are {colored(j.flux_units, "cyan")}')

                if (length_scale_factor*unit_prod/j.flux_units).dimensionless:
                    pass
                elif (1/length_scale_factor*unit_prod/j.flux_units).dimensionless:
                    length_scale_factor = 1/length_scale_factor
                else:
                    raise Exception("Inconsitent units!")

                Print('Adjusted flux with the length scale factor ' +
                      colored("%f [%s]"%(length_scale_factor.magnitude,str(length_scale_factor.units)), "cyan") + ' to match units.\n')

                prod *= length_scale_factor.magnitude
                total_scaling *= length_scale_factor.magnitude
                unit_prod *= length_scale_factor.units*1
                setattr(j, 'length_scale_factor', length_scale_factor)

            # if units are consistent in dimensionality but not magnitude, adjust values
            if j.flux_units != unit_prod:
                unit_scaling = unit_prod.to(j.flux_units).magnitude
                total_scaling *= unit_scaling
                prod *= unit_scaling
                Print(('\nThe flux, %s, has units '%j.flux_name + colored(unit_prod, "red") +
                    "...the desired units for this flux are " + colored(j.flux_units, "cyan")))
                Print('Adjusted value of flux by ' + colored("%f"%unit_scaling, "cyan") + ' to match units.\n')
                setattr(j, 'unit_scaling', unit_scaling)
            else:
                setattr(j, 'unit_scaling', 1)

            setattr(j, 'total_scaling', total_scaling)

            # adjust sign+stoich if necessary
            prod *= j.signed_stoich

            # multiply by appropriate integration measure and test function
            if j.flux_dimensionality[0] < j.flux_dimensionality[1]:
                form_key = 'B'
            else:
                form_key = 'R'
            #prod = prod*sp.v*j.int_measure
            dolfin_flux = prod*j.int_measure

            setattr(j, 'dolfin_flux', dolfin_flux)

            BRform = -prod*sp.v*j.int_measure # by convention, terms are all defined as if they were on the lhs of the equation e.g. F(u;v)=0
            self.Forms.add(stubs.model_assembly.Form(BRform, sp, form_key, flux_name=j.name))


    def assemble_diffusive_fluxes(self):
        min_dim = min(self.cc.get_property('dimensionality').values())
        max_dim = max(self.cc.get_property('dimensionality').values())
        dT = self.dT

        for sp_name, sp in self.sc.items:
            if sp.is_in_a_reaction:
                if self.solver_system.nonlinear_solver.method in ['picard', 'IMEX']:
                    u = sp.u['t']
                elif self.solver_system.nonlinear_solver.method == 'newton':
                    u = sp.u['u']
                un = sp.u['n']
                v = sp.v
                D = sp.D

                if sp.dimensionality == max_dim:
                    dx = sp.compartment.dx
                    Dform = D*d.inner(d.grad(u), d.grad(v)) * dx
                    self.Forms.add(stubs.model_assembly.Form(Dform, sp, 'D'))
                elif sp.dimensionality < max_dim:
                    if self.solver_system.ignore_surface_diffusion:
                        dx=sp.compartment.dP
                    else:
                        dx = sp.compartment.dx
                        Dform = D*d.inner(d.grad(u), d.grad(v)) * dx
                        self.Forms.add(stubs.model_assembly.Form(Dform, sp, 'D'))

                # time derivative
                Mform_u = u/dT * v * dx
                Mform_un = -un/dT * v * dx
                self.Forms.add(stubs.model_assembly.Form(Mform_u, sp, "Mu"))
                self.Forms.add(stubs.model_assembly.Form(Mform_un, sp, "Mun"))

            else:
                Print("Species %s is not in a reaction?" %  sp_name)

    def set_allow_extrapolation(self):
        for comp_name in self.u.keys():
            ucomp = self.u[comp_name]
            for func_key in ucomp.keys():
                if func_key != 't': # trial function by convention
                    self.u[comp_name][func_key].set_allow_extrapolation(True)

    def sort_forms(self):
        """
        Organizes forms based on solution method. E.g. for picard iterations we
        split the forms into a bilinear and linear component, for Newton we
        simply solve F(u;v)=0.
        """
        comp_list = [self.cc[key] for key in self.u.keys()]
        self.split_forms = ddict(dict)
        form_types = set([f.form_type for f in self.Forms.form_list])

        if self.solver_system.nonlinear_solver.method == 'picard':
            raise Exception("Picard functionality needs to be reviewed")
            # Print("Splitting problem into bilinear and linear forms for picard iterations: a(u,v) == L(v)")
            # for comp in comp_list:
            #     comp_forms = [f.dolfin_form for f in self.Forms.select_by('compartment_name', comp.name)]
            #     self.a[comp.name] = d.lhs(sum(comp_forms))
            #     self.L[comp.name] = d.rhs(sum(comp_forms))
            #     problem = d.LinearVariationalProblem(self.a[comp.name],
            #                                          self.L[comp.name], self.u[comp.name]['u'], [])
            #     self.linear_solver[comp.name] = d.LinearVariationalSolver(problem)
            #     p = self.linear_solver[comp.name].parameters
            #     p['linear_solver'] = self.solver_system.linear_solver.method
            #     if type(self.solver_system.linear_solver) == stubs.solvers.DolfinKrylovSolver:
            #         p['krylov_solver'].update(self.solver_system.linear_solver.__dict__)
            #         p['krylov_solver'].update({'nonzero_initial_guess': True}) # important for time dependent problems

        elif self.solver_system.nonlinear_solver.method == 'newton':
            Print("Formulating problem as F(u;v) == 0 for newton iterations")
            for comp in comp_list:
                comp_forms = [f.dolfin_form for f in self.Forms.select_by('compartment_name', comp.name)]
                self.F[comp.name] = sum(comp_forms)
                J = d.derivative(self.F[comp.name], self.u[comp.name]['u'])

                problem = d.NonlinearVariationalProblem(self.F[comp.name], self.u[comp.name]['u'], [], J)

                self.nonlinear_solver[comp.name] = d.NonlinearVariationalSolver(problem)
                p = self.nonlinear_solver[comp.name].parameters
                p['nonlinear_solver'] = 'newton'
                p['newton_solver'].update(self.solver_system.nonlinear_dolfin_solver_settings)
                p['newton_solver']['krylov_solver'].update(self.solver_system.linear_dolfin_solver_settings)
                p['newton_solver']['krylov_solver'].update({'nonzero_initial_guess': True}) # important for time dependent problems

        elif self.solver_system.nonlinear_solver.method == 'IMEX':
            raise Exception("IMEX functionality needs to be reviewed")
#            Print("Keeping forms separated by compartment and form_type for IMEX scheme.")
#            for comp in comp_list:
#                comp_forms = self.Forms.select_by('compartment_name', comp.name)
#                for form_type in form_types:
#                    self.split_forms[comp.name][form_type] = sum([f.dolfin_form for f in comp_forms if f.form_type==form_type])

    #===============================================================================
    # Model - Solving
    #=============================================================================== 
    def solve_single_timestep(self, plot_period=1):
        # Solve using specified multiphysics scheme 
        if self.solver_system.multiphysics_solver.method == "iterative":
            self.iterative_mpsolve()
        else:
            raise Exception("I don't know what operator splitting scheme to use")

        # post processing
        self.post_process()
        if (self.idx % plot_period == 0 or self.t >= self.final_t) and plot_period!=0:
            self.plot_solution()

        # if we've reached final time
        end_simulation = self.t >= self.final_t

        return end_simulation

    def solve(self, plot_period=1):
        # Initialize
        #self.init_solutions_and_plots()

        self.stopwatch("Total simulation")
        # Time loop
        while True:
            end_simulation = self.solve_single_timestep(plot_period)

            if end_simulation:
                break

        self.stopwatch("Total simulation", stop=True)
        Print("Solver finished with %d total time steps." % self.idx)

    def set_time(self, t, dt=None):
        if not dt:
            dt = self.dt
        else:
            Print("dt changed from %f to %f" % (self.dt, dt))
        if t != self.t:
            Print("Time changed from %f to %f" % (self.t, t))
        self.t = t
        self.T.assign(t)
        self.dt = dt
        self.dT.assign(dt)

    def check_dt_resets(self):
        """
        Checks to see if the size of a full-time step would pass a "reset dt"
        checkpoint. At these checkpoints dt is reset to some value
        (e.g. to force smaller sampling during fast events)
        """
        # check that there are reset times specified
        if self.solver_system.adjust_dt is None or len(self.solver_system.adjust_dt) == 0:
            return

        # if last time-step we passed a reset dt checkpoint then reset it now
        if self.reset_dt:
            new_dt = self.solver_system.adjust_dt[0][1]
            fancy_print(f"(!!!) Adjusting time-step (dt = {self.dt} -> {new_dt}) to match config specified value", format_type='log')
            self.set_time(self.t, dt = new_dt)
            del(self.solver_system.adjust_dt[0])
            self.reset_dt = False
            return

        # check if we pass a reset dt checkpoint
        t0 = self.t # original time
        potential_t = t0 + self.dt # the final time if dt is not reset
        next_reset_time = self.solver_system.adjust_dt[0][0] # the next time value to reset dt at
        next_reset_dt = self.solver_system.adjust_dt[0][1] # the next value of dt to reset to
        if next_reset_time<t0:
            raise Exception("Next reset time is less than time at beginning of time-step.")
        if t0 < next_reset_time <= potential_t: # check if the full time-step would pass a reset dt checkpoint
            new_dt = max([next_reset_time - t0, next_reset_dt]) # this is needed otherwise very small time-steps might be taken which wont converge
            fancy_print("(!!!) Adjusting time-step (dt = {self.dt} -> {new_dt}) to avoid passing reset dt checkpoint", format_type='log_important')
            self.set_time(self.t, dt=new_dt)
            # set a flag to change dt to the config specified value
            self.reset_dt = True

    def check_dt_pass_tfinal(self):
        """
        Check that t+dt is not > t_final
        """
        potential_t = self.t + self.dt
        if potential_t > self.final_t:
            new_dt = self.final_t - self.t 
            fancy_print("(!!!) Adjusting time-step (dt = {self.dt} -> {new_dt}) to avoid passing final time", format_type='log')
            self.set_time(self.t, dt=new_dt)

    def forward_time_step(self, dt_factor=1):

        self.dT.assign(float(self.dt*dt_factor))
        self.t = float(self.t+self.dt*dt_factor)
        self.T.assign(self.t)
        self.update_time_dependent_parameters(dt=self.dt*dt_factor)
        Print("t: %f , dt: %f" % (self.t, self.dt*dt_factor))

    def stopwatch(self, key, stop=False, color='cyan'):
        if key not in self.timers.keys():
            self.timers[key] = d.Timer()
        if not stop:
            self.timers[key].start()
        else:
            elapsed_time = self.timers[key].elapsed()[0]
            time_str = str(elapsed_time)[0:8]
            if color:
                time_str = colored(time_str, color)
            Print("%s finished in %s seconds" % (key,time_str))
            self.timers[key].stop()
            self.timings[key].append(elapsed_time)
            return elapsed_time

    def update_time_dependent_parameters(self, t=None, t0=None, dt=None):
        """ Updates all time dependent parameters. Time-dependent parameters are 
        either defined either symbolically or through a data file, and each of 
        these can either be defined as a direct function of t, p(t), or a 
        "pre-integrated expression", \int_{0}^{t} P(x) dx, which allows for 
        exact integration when the expression the parameter appears in is purely
        time-dependent.
        
        Args:
            t (float, optional): Current time in the simulation to update parameters to 
            t0 (float, optional): Previous time value (i.e. t0 == t-dt)
            dt (float, optional): Time step
        
        Raises:
            Exception: Description
        """
        if t is None:
            t = self.t
        if t0 is not None and dt is not None:
            raise Exception("Specify either t0 or dt, not both.")
        elif t0 is not None:
            dt = t-t0
        elif dt is not None:
            t0 = t-dt
        if t0 is not None:
            if t0<0 or dt<=0:
                raise Exception("Either t0<0 or dt<=0, is this the desired behavior?")

        # Update time dependent parameters
        for param_name, param in self.pc.items:
            # check to make sure a parameter wasn't assigned a new value more than once
            value_assigned = 0
            if not param.is_time_dependent:
                continue
            # use value by evaluating symbolic expression
            if param.sym_expr and not param.preint_sym_expr:
                newValue = float(param.sym_expr.subs({'t': t}).evalf())
                value_assigned += 1
                print(f"Parameter {param_name} assigned by symbolic expression")

            # calculate a preintegrated expression by subtracting previous value
            # and dividing by time-step
            if param.sym_expr and param.preint_sym_expr:
                if t0 is None:
                    raise Exception("Must provide a time interval for"\
                                    "pre-integrated variables.")
                a = param.preint_sym_expr.subs({'t': t0}).evalf()
                b = param.preint_sym_expr.subs({'t': t}).evalf()
                newValue = float((b-a)/dt)
                value_assigned += 1
                print(f"Parameter {param_name} assigned by preintegrated symbolic expression")

            # if parameter is given by data
            if param.sampling_data is not None and param.preint_sampling_data is None:
                data = param.sampling_data
                # We never want time to extrapolate beyond the provided data.
                if t<data[0,0] or t>data[-1,0]:
                    raise Exception("Parameter cannot be extrapolated beyond"\
                                    "provided data.")
                # Just in case... return a nan if value is outside of bounds
                newValue = float(np.interp(t, data[:,0], data[:,1],
                                     left=np.nan, right=np.nan))
                value_assigned += 1
                print(f"Parameter {param_name} assigned by sampling data")

            # if parameter is given by data and it has been pre-integrated
            if param.sampling_data is not None and param.preint_sampling_data is not None:
                int_data = param.preint_sampling_data
                a = np.interp(t0, int_data[:,0], int_data[:,1],
                                     left=np.nan, right=np.nan)
                b = np.interp(t, int_data[:,0], int_data[:,1],
                                     left=np.nan, right=np.nan)
                newValue = float((b-a)/dt)
                value_assigned += 1
                print(f"Parameter {param_name} assigned by preintegrated sampling data")

            if value_assigned != 1:
                raise Exception("Either a value was not assigned or more than"\
                                "one value was assigned to parameter %s" % param.name)

            Print('%f assigned to time-dependent parameter %s'
                  % (newValue, param.name))

            if np.isnan(newValue):
                raise ValueError(f"Warning! Parameter {param_name} is a NaN.")

            param.value = newValue
            param.dolfin_constant.assign(newValue)
            self.params[param_name].append((t,newValue))


    def reset_timestep(self, comp_list=[]):
        """
        Resets the time back to what it was before the time-step. Optionally, input a list of compartments
        to have their function values reset (['n'] value will be assigned to ['u'] function).
        """
        self.set_time(self.t - self.dt, self.dt*self.solver_system.nonlinear_solver.dt_decrease_dt_factor)
        Print("Resetting time-step and decreasing step size")
        for comp_name in comp_list:
            self.u[comp_name]['u'].assign(self.u[comp_name]['n'])
            Print("Assigning old value of u to species in compartment %s" % comp_name)

    def update_solution_boundary_to_volume(self):
        for comp_name in self.cc.keys:
            for key in self.u[comp_name].keys():
                if key[0:2] == 'v_': # fixme
                    d.LagrangeInterpolator.interpolate(self.u[comp_name][key], self.u[comp_name]['u'])
                    parent_comp_name = key[2:]
                    Print("Projected values from surface %s to volume %s" % (comp_name, parent_comp_name))

    def update_solution_volume_to_boundary(self):
        for comp_name in self.cc.keys:
            for key in self.u[comp_name].keys():
                if key[0:2] == 'b_': # fixme
                    #self.u[comp_name][key].interpolate(self.u[comp_name]['u'])
                    d.LagrangeInterpolator.interpolate(self.u[comp_name][key], self.u[comp_name]['u'])
                    sub_comp_name = key[2:]
                    Print("Projected values from volume %s to surface %s" % (comp_name, sub_comp_name))

    def boundary_reactions_forward(self, dt_factor=1, bcs=[]):
        self.stopwatch("Boundary reactions forward")

        # solve boundary problem(s)
        for comp_name, comp in self.cc.items:
            if comp.dimensionality < self.cc.max_dim:
                self.nonlinear_solve(comp_name, dt_factor=dt_factor)
        self.stopwatch("Boundary reactions forward", stop=True)

    def volume_reactions_forward(self, dt_factor=1, bcs=[], time_step=True):
        self.stopwatch("Volume reactions forward")

        # solve volume problem(s)
        for comp_name, comp in self.cc.items:
            if comp.dimensionality == self.cc.max_dim:
                self.nonlinear_solve(comp_name, dt_factor=dt_factor)
        self.stopwatch("Volume reactions forward", stop=True)

    def nonlinear_solve(self, comp_name, dt_factor=1.0):
        """
        A switch for choosing a nonlinear solver
        """

        if self.solver_system.nonlinear_solver.method == 'newton':
            self.stopwatch("Newton's method [%s]" % comp_name)
            self.nl_idx[comp_name], self.success[comp_name] = self.nonlinear_solver[comp_name].solve()
            self.stopwatch("Newton's method [%s]" % comp_name, stop=True)
            Print(f"{self.nl_idx[comp_name]} Newton iterations required for convergence on compartment {comp_name}.")
        # elif self.solver_system.nonlinear_solver.method == 'picard':
        #     self.stopwatch("Picard iteration method [%s]" % comp_name)
        #     self.nl_idx[comp_name], self.success[comp_name] = self.picard_loop(comp_name, dt_factor=dt_factor)
        #     self.stopwatch("Picard iteration method [%s]" % comp_name)
        #     Print(f"{self.nl_idx[comp_name]} Newton iterations required for convergence on compartment {comp_name}.")
        else:
            raise ValueError("Unknown nonlinear solver method")

    def update_solution(self, ukeys=['k', 'n']):
        """
        After finishing a time step, assign all the most recently computed solutions as 
        the solutions for the previous time step.
        """
        for key in self.u.keys():
            for ukey in ukeys:
                self.u[key][ukey].assign(self.u[key]['u'])

    def adjust_dt(self):
        ## check if time step should be changed
        #if all([x <= self.solver_system.nonlinear_solver.min_nonlinear for x in self.nl_idx.values()]):
        #    self.set_time(self.t, dt=self.dt*self.solver_system.nonlinear_solver.dt_increase_factor)
        #    Print("Increasing step size")
        #elif any([x > self.solver_system.nonlinear_solver.max_nonlinear for x in self.nl_idx.values()]):
        #    self.set_time(self.t, dt=self.dt*self.solver_system.nonlinear_solver.dt_decrease_factor)
        #    Print("Decreasing step size")

        # check if time step should be changed based on number of multiphysics iterations
        if self.mpidx <= self.solver_system.multiphysics_solver.min_multiphysics:
            self.set_time(self.t, dt=self.dt*self.solver_system.multiphysics_solver.dt_increase_factor)
            Print("Increasing step size")
        elif self.mpidx > self.solver_system.multiphysics_solver.max_multiphysics:
            self.set_time(self.t, dt=self.dt*self.solver_system.multiphysics_solver.dt_decrease_factor)
            Print("Decreasing step size")

    def compute_stopping_conditions(self, norm_type=np.inf):

        for comp_name in self.u.keys():
            # dolfin norm() is not really robust so we get vectors and use numpy.linalg.norm
            #Fabs = d.norm(d.assemble(self.F[comp_name]), norm_type)
            #udiff = self.u[comp_name]['u'] - self.u[comp_name]['k']
            #udiff_abs = d.norm(udiff, norm_type)
            #udiff_rel = udiff_abs/d.norm(self.u[comp_name]['u'], norm_type)

            Fvec = d.assemble(self.F[comp_name]).get_local()

            Fabs = np.linalg.norm(Fvec, ord=norm_type)

            self.stopping_conditions['F_abs'].update({comp_name: Fabs})

        for sp_name, sp in self.sc.items:
            uvec = self.dolfin_get_function_values(sp, ukey='u')
            ukvec = self.dolfin_get_function_values(sp, ukey='k')
            udiff = uvec - ukvec

            udiff_abs = np.linalg.norm(udiff, ord=norm_type)
            udiff_rel = udiff_abs/np.linalg.norm(uvec, ord=norm_type)

            self.stopping_conditions['udiff_abs'].update({sp_name: udiff_abs})
            self.stopping_conditions['udiff_rel'].update({sp_name: udiff_rel})

    def iterative_mpsolve(self, bcs=[]):
        """
        Iterate between boundary and volume problems until convergence
        """
        Print('\n\n\n')
        self.idx += 1
        self.check_dt_resets()      # check if there is a manually prescribed time-step size
        self.check_dt_pass_tfinal() # check that we don't pass tfinal
        fancy_print(f'Beginning time-step {self.idx} [time={self.t}, dt={self.dt}]', format_type='timestep')

        self.stopwatch("Total time step") # start a timer for the total time step

        self.forward_time_step() # march forward in time and update time-dependent parameters

        self.mpidx = 0
        while True: 
            self.mpidx += 1
            fancy_print(f'Multiphysics iteration {self.mpidx} for time-step {self.idx} [time={self.t}]', format_type='solverstep')
            # solve volume problem(s)
            self.volume_reactions_forward()
            self.update_solution_volume_to_boundary()
            # solve boundary problem(s)
            self.boundary_reactions_forward()
            self.update_solution_boundary_to_volume()
            # decide whether to stop iterations
            self.compute_stopping_conditions()
            if self.solver_system.multiphysics_solver.eps_Fabs is not None:
                max_comp_name, max_Fabs = max(self.stopping_conditions['F_abs'].items(), key=operator.itemgetter(1))
                if all([x<self.solver_system.multiphysics_solver.eps_Fabs for x in self.stopping_conditions['F_abs'].values()]):
                    fancy_print(f"All F_abs are below tolerance ({self.solver_system.multiphysics_solver.eps_Fabs:.4e}", format_type='log_important')
                    fancy_print(f"Max F_abs is {max_Fabs:.4e} from compartment {max_comp_name}", format_type='log_important')
                    fancy_print(f"Exiting multiphysics loop ({self.mpidx} iterations)", format_type='log_important')
                    break
                else: 
                    #max_comp_name, max_Fabs = max(self.stopping_conditions['F_abs'].items(), key=operator.itemgetter(1))
                    #color_print(f"{'One or more F_abs are above tolerance. Max F_abs is from compartment '+max_comp_name+': ': <40} {max_Fabs:.4e}", color='green')
                    fancy_print(f"One or more F_abs are above tolerance ({self.solver_system.multiphysics_solver.eps_Fabs:.4e}", format_type='log')
                    fancy_print(f"Max F_abs is {max_Fabs:.4e} from compartment {max_comp_name}", format_type='log')

            if self.solver_system.multiphysics_solver.eps_udiff_abs is not None:
                max_sp_name, max_udiffabs = max(self.stopping_conditions['udiff_abs'].items(), key=operator.itemgetter(1))
                if all([x<self.solver_system.multiphysics_solver.eps_udiff_abs for x in self.stopping_conditions['udiff_abs'].values()]):
                    #color_print(f"All udiff_abs are below tolerance, {self.solver_system.multiphysics_solver.eps_udiff_abs}." \
                    #              f" Exiting multiphysics loop ({self.mpidx} iterations).\n", color='green')
                    fancy_print(f"All udiff_abs are below tolerance ({self.solver_system.multiphysics_solver.eps_udiff_abs:.4e})", format_type='log_important')
                    fancy_print(f"Max udiff_abs is {max_udiffabs:.4e} from species {max_sp_name}", format_type='log_important')
                    fancy_print(f"Exiting multiphysics loop ({self.mpidx} iterations)", format_type='log_important')
                    break
                else: 
                    #color_print(f"{'One or more udiff_abs are above tolerance. Max udiff_abs is from species '+max_sp_name+': ': <40} {max_udiffabs:.4e}", color='green')
                    fancy_print(f"One or more udiff_abs are above tolerance ({self.solver_system.multiphysics_solver.eps_udiff_abs:.4e}", format_type='log')
                    fancy_print(f"Max udiff_abs is {max_udiffabs:.4e} from species {max_sp_name}", format_type='log')

            if self.solver_system.multiphysics_solver.eps_udiff_rel is not None:
                max_sp_name, max_udiffrel = max(self.stopping_conditions['udiff_rel'].items(), key=operator.itemgetter(1))
                if all([x<self.solver_system.multiphysics_solver.eps_udiff_rel for x in self.stopping_conditions['udiff_rel'].values()]):
                    #color_print(f"All udiff_rel are below tolerance, {self.solver_system.multiphysics_solver.eps_udiff_rel}." \
                    #              f" Exiting multiphysics loop ({self.mpidx} iterations).\n", color='green')
                    fancy_print(f"All udiff_rel are below tolerance ({self.solver_system.multiphysics_solver.eps_udiff_rel:.4e})", format_type='log_important')
                    fancy_print(f"Max udiff_rel is {max_udiffrel:.4e} from species {max_sp_name}", format_type='log_important')
                    fancy_print(f"Exiting multiphysics loop ({self.mpidx} iterations)", format_type='log_important')
                    break
                else: 
                    max_sp_name, max_udiffrel = max(self.stopping_conditions['udiff_rel'].items(), key=operator.itemgetter(1))
                    #color_print(f"{'One or more udiff_rel are above tolerance. Max udiff_rel is from species '+max_sp_name+': ': <40} {max_udiffrel:.4e}", color='green')
                    fancy_print(f"One or more udiff_rel are above tolerance ({self.solver_system.multiphysics_solver.eps_udiff_rel:.4e}", format_type='log')
                    fancy_print(f"Max udiff_rel is {max_udiffabs:.4e} from species {max_sp_name}", format_type='log')

            # update previous (iteration) solution 
            self.update_solution(ukeys=['k'])


        self.update_solution() # assign most recently computed solution to "previous solution"
        self.adjust_dt() # adjusts dt based on number of nonlinear iterations required
        self.stopwatch("Total time step", stop=True, color='cyan')


#===============================================================================
#===============================================================================
# Nonlinear solvers:
#  - Timestep
#  - Update time dependent parameters
#  - Solve
#  - Assign new solution to old solution unless otherwise stated
#===============================================================================
#===============================================================================



#     def picard_loop(self, comp_name, dt_factor=1, bcs=[]):
#         """
#         Continue picard iterations until a specified tolerance or count is
#         reached.
#         """
#         self.stopwatch("Picard loop [%s]" % comp_name)
#         t0 = self.t
#         self.forward_time_step(dt_factor=dt_factor) # increment time afterwards
#         self.pidx = 0 # count the number of picard iterations
#         success = True

#         # main loop
#         while True:
#             self.pidx += 1
#             #linear_solver_settings = self.config.dolfin_krylov_solver

#             # solve
#             self.linear_solver[comp_name].solve()
#             #d.solve(self.a[comp_name]==self.L[comp_name], self.u[comp_name]['u'], bcs, solver_parameters=linear_solver_settings)

#             # update temporary value of u
#             self.data.compute_error(self.u, comp_name, self.solver_system.nonlinear_solver.picard_norm)
#             self.u[comp_name]['k'].assign(self.u[comp_name]['u'])

#             # Exit if error tolerance or max iterations is reached
#             Print('Linf norm (%s) : %f ' % (comp_name, self.data.errors[comp_name]['Linf']['abs'][-1]))
#             if self.data.errors[comp_name]['Linf']['abs'][-1] < self.solver_system.nonlinear_solver.absolute_tolerance:
#                 #print("Norm (%f) is less than linear_abstol (%f), exiting picard loop." %
#                  #(self.data.errors[comp_name]['Linf'][-1], self.config.solver['linear_abstol']))
#                 break
# #            if self.data.errors[comp_name]['Linf']['rel'][-1] < self.config.solver['linear_reltol']:
# #                print("Norm (%f) is less than linear_reltol (%f), exiting picard loop." %
# #                (self.data.errors[comp_name]['Linf']['rel'][-1], self.config.solver['linear_reltol']))
# #                break

#             if self.pidx >= self.solver_system.nonlinear_solver.maximum_iterations:
#                 Print("Max number of picard iterations reached (%s), exiting picard loop with abs error %f." %
#                 (comp_name, self.data.errors[comp_name]['Linf']['abs'][-1]))
#                 success = False
#                 break

#         self.stopwatch("Picard loop [%s]" % comp_name, stop=True)

#         Print("%d Picard iterations required for convergence." % self.pidx)
#         return self.pidx, success


    #===============================================================================
    # Model - Post-processing
    #===============================================================================
    def init_solutions_and_plots(self):
        self.data.init_solution_files(self.sc, self.config)
        self.data.store_solution_files(self.u, self.t, self.config)
        self.data.compute_statistics(self.u, self.t, self.dt, self.sc, self.pc, self.cc, self.fc, self.nl_idx)
        self.data.init_plot(self.config, self.sc, self.fc)

    def post_process(self):
        self.data.compute_statistics(self.u, self.t, self.dt, self.sc, self.pc, self.cc, self.fc, self.nl_idx)
        self.data.compute_probe_values(self.u, self.sc)
        self.data.output_pickle()
        self.data.output_csv()

    def plot_solution(self):
        self.data.store_solution_files(self.u, self.t, self.config)
        self.data.plot_parameters(self.config)
        self.data.plot_solutions(self.config, self.sc)
        self.data.plot_fluxes(self.config)
        self.data.plot_solver_status(self.config)


    #===============================================================================
    # Model - Data manipulation
    #===============================================================================
    # get the values of function u from subspace idx of some mixed function space, V
    @staticmethod
    def dolfin_get_dof_indices(species):#V, species_idx):
        """
        Returned indices are *local* to the CPU (not global)
        function values can be returned e.g.
        indices = dolfin_get_dof_indices(V,species_idx)
        u.vector().get_local()[indices]
        """
        if species.dof_map is not None:
            return species.dof_map
        #V           = sp.compartment.V
        V           = species.V
        species_idx = species.dof_index

        # if V.num_sub_spaces() > 1:
        #     indices = np.array(V.sub(species_idx).dofmap().dofs())
        # else:
        indices = np.array(V.dofmap().dofs())
        first_idx, last_idx = V.dofmap().ownership_range() # indices that this CPU owns

        return indices-first_idx # subtract index offset to go from global -> local indices

    @staticmethod
    def reduce_vector(u):
        """
        comm.allreduce() only works when the incoming vectors all have the same length. We use comm.Gatherv() to gather vectors
        with different lengths
        """
        sendcounts = np.array(comm.gather(len(u), root)) # length of vectors being sent by workers
        if rank == root:
            print("reduceVector(): CPUs sent me %s length vectors, total length: %d"%(sendcounts, sum(sendcounts)))
            recvbuf = np.empty(sum(sendcounts), dtype=float)
        else:
            recvbuf = None

        comm.Gatherv(sendbuf=u, recvbuf=(recvbuf, sendcounts), root=root)

        return recvbuf

    def dolfin_get_function_values(self, sp, ukey='u'):
        """
        Returns the values from a sub-function of a VectorFunction. When run
        in parallel this will *not* double-count overlapping vertices which
        are shared by multiple CPUs (a simple call to 
        u.sub(species_idx).compute_vertex_values() will double-count...)
        """
        # Get the VectorFunction
        #V           = sp.compartment.V
        #species_idx = sp.dof_index
        indices     = self.dolfin_get_dof_indices(sp)
        uvec        = sp.u[ukey].vector().get_local()[indices]
    
        return uvec

    def dolfin_get_function_values_at_point(self, sp, coord):
        """
        Returns the values of a dolfin function at the specified coordinate 
        :param dolfin.function.function.Function u: Function to extract values from
        :param tuple coord: tuple of floats indicating where in space to evaluate u e.g. (x,y,z)
        :param int species_idx: index of species
        :return: list of values at point. If species_idx is not specified it will return all values
        """
        return sp.u['u'](coord)
        # u = self.u[sp.compartment_name]['u']
        # if sp.compartment.V.num_sub_spaces() == 0:
        #     return u(coord)
        # else:
        #     species_idx = sp.dof_index
        #     return u(coord)[species_idx]

    def dolfin_set_function_values(self, sp, ukey, unew):
        """
        d.assign(uold, unew) works when uold is a subfunction
        uold.assign(unew) does not (it will replace the entire function)
        """
        if isinstance(unew, d.Expression):
            uinterp = d.interpolate(unew, sp.V)
            d.assign(sp.u[ukey], uinterp)
        elif isinstance(unew, (float,int)):
            uinterp = d.interpolate(d.Constant(unew), sp.V)
            d.assign(sp.u[ukey], uinterp)
        else:
            # unew is a vector with the same length as u
            raise NotImplementedError
            # #u = self.u[ukey][sp.compartment_name][ukey]
            # u = self.cc[sp.compartment_name].u[ukey]

            # #indices = self.dolfin_get_dof_indices(sp)
            # indices = sp.dof_map
            # uvec    = u.vector()
            # values  = uvec.get_local()
            # values[indices] = unew

            # uvec.set_local(values)
            # uvec.apply('insert')

    # def assign_initial_conditions(self):
    #     ukeys = ['k', 'n', 'u']
    #     for sp_name, sp in self.sc.items:
    #         comp_name = sp.compartment_name
    #         for ukey in ukeys:
    #             self.dolfin_set_function_values(sp, ukey, sp.initial_condition)
    #         if rank==root: print("Assigned initial condition for species %s" % sp.name)

    #     # project to boundary/volume functions
    #     self.update_solution_volume_to_boundary()
    #     self.update_solution_boundary_to_volume()

    # def dolfinFindClosestPoint(mesh, coords):
    #     """
    #     Given some point and a mesh, returns the coordinates of the nearest vertex
    #     """
    #     p = d.Point(coords)
    #     L = list(d.vertices(mesh))
    #     distToVerts = [np.linalg.norm(p.array() - x.midpoint().array()) for x in L]
    #     minDist = min(distToVerts)
    #     minIdx = distToVerts.index(minDist) # returns the local index (wrt the cell) of the closest point

    #     closestPoint = L[minIdx].midpoint().array()

    #     if size > 1:
    #         min_dist_global, min_idx = comm.allreduce((minDist,rank), op=pyMPI.MINLOC)

    #         if rank == min_idx:
    #             comm.Send(closestPoint, dest=root)

    #         if rank == root:
    #             comm.Recv(closestPoint, min_idx)
    #             print("CPU with rank %d has the closest point to %s: %s. The distance is %s" % (min_idx, coords, closestPoint, min_dist_global))
    #             return closestPoint, min_dist_global
    #     else:
    #         return closestPoint, minDist

# # # Monkey patching
#     @staticmethod
#     def sub_patch(func, idx):
#         "This is just a hack patch to allow us to refer to a function/functionspace with no subspaces using .sub(0)"
#         if func.num_sub_spaces() <= 1 and idx == 0:
#             return func
#         else:
#             sub_func = func.sub(idx)
#             Model.apply_patch(sub_func, )
#             sub_func._sub = sub_func.sub
#             sub_func.sub = Model.sub_patch.__get__(sub_func, d.Function)
#             return func.sub(idx)
#     @staticmethod
#     def apply_patch(func, base_class):
#         func._sub = func.sub
#         func.sub = Model.sub_patch.__get__(func, base_class)
            
#     @staticmethod    
#     def Function(function_space):
#         func = d.Function(function_space)

#         func._sub = func.sub
#         func.sub = Model.sub_patch.__get__(func, d.Function)
#         return func

#     @staticmethod
#     def VectorFunctionSpace(mesh, family, degree, dim):
#         if dim > 1:
#             func_space = d.VectorFunctionSpace(mesh, family, degree, dim=dim)
#             return func_space
#         elif dim == 1:
#             func_space = d.FunctionSpace(mesh, family, degree)
        
#             func_space._sub = func_space.sub
#             func_space.sub = Model.sub_patch.__get__(func_space, d.FunctionSpace)
#             return func_space