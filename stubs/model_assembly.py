"""
Classes for parameters, species, compartments, reactions, fluxes, and forms
Model class contains functions to efficiently solve a system
"""
import dataclasses
import numbers
import sys
import logging
from collections import OrderedDict as odict
from dataclasses import dataclass
from pprint import pformat
from textwrap import wrap
from typing import Any, Union

import dolfin as d
import numpy as np
import numpy.typing as npt
import pandas
import pint
import sympy as sym
import ufl
from cached_property import cached_property
from sympy import Symbol, integrate
from sympy.parsing.sympy_parser import parse_expr
from tabulate import tabulate

from . import common
from .common import pint_quantity_to_unit, pint_unit_to_quantity, sub
from .config import global_settings as gset
from .units import unit


__all__ = [
    "empty_sbmodel",
    "sbmodel_from_locals",
    "Compartment",
    "Parameter",
    "Reaction",
    "Species",
    "sbmodel_from_locals",
]

comm = d.MPI.comm_world
rank = comm.rank
size = comm.size
root = 0

logger = logging.getLogger(__name__)


def _np_smart_hstack(
    x1: Union[list, npt.ArrayLike], x2: Union[list, npt.ArrayLike]
) -> npt.ArrayLike:
    """
    Quality of life function. Converts two (N,) numpy arrays or two lists into a
    (Nx2) array

    Params:
        x1: First array
        x2: Second array

    Returns:
        The stacked array

    Examples:

        .. highlight:: python
        .. code-block:: python

            a_list = [1,2,3,4]
            a_np_array = np.array([x**2 for x in a_list])
            _np_smart_hstack(a_list, a_np_array)
    """
    assert len(x1) == len(x2)  # confirm same size
    if isinstance(x1, list):
        x1 = np.array(x1)
    if isinstance(x2, list):
        x2 = np.array(x2)
    return np.hstack([x1.reshape(-1, 1), x2.reshape(-1, 1)])


# ====================================================
# ====================================================
# Base Classes
# ====================================================
# ====================================================


class ObjectContainer:
    """
    Parent class containing general methods used by all "containers"
    """

    def __init__(self, ObjectClass):  # df=None, Dict=None):
        self.Dict = odict()
        self._ObjectClass = ObjectClass
        self.properties_to_print = []  # properties to print

    # =====================================================================
    # ObjectContainer - General methods
    # ---------------------------------------------------------------------
    # General properties/methods to emulate some dictionary-like structure
    # + add some misc useful structure
    # =====================================================================

    @property
    def size(self):
        "Size of ObjectContainer"
        return len(self.Dict)

    @property
    def items(self):
        return self.Dict.items()

    @property
    def values(self):
        return self.Dict.values()

    def __iter__(self):
        return iter(self.Dict.values())

    @property
    def keys(self):
        return self.Dict.keys()

    @property
    def indices(self):
        return enumerate(self.keys)

    def to_dicts(self):
        "Returns a list of dicts that can be used to recreate the ObjectContainer"
        return [obj.to_dict() for obj in self.values]

    def __getitem__(self, key):
        "syntactic sugar to allow: objcontainer[key] = objcontainer[key]"
        return self.Dict[key]

    def __setitem__(self, key, newvalue):
        "syntactic sugar to allow: objcontainer[key] = objcontainer[key]"
        self.Dict[key] = newvalue

    def add(self, *data):
        if len(data) == 1:
            data = data[0]
            # Adding in the ObjectInstance directly
            if isinstance(data, self._ObjectClass):
                self[data.name] = data
            # Adding in an iterable of ObjectInstances
            else:
                # check if input is an iterable and if so add it item by item
                try:
                    iter(data)
                except Exception:
                    raise TypeError(
                        "Data being added to ObjectContainer must be either the "
                        "ObjectClass or an iterator."
                    )
                else:
                    if isinstance(data, dict):
                        for obj_name, obj in data.items():
                            self[obj_name] = obj
                    elif all([isinstance(obj, self._ObjectClass) for obj in data]):
                        for obj in data:
                            self[obj.name] = obj
                    else:
                        raise Exception("Could not add data to ObjectContainer")
        # Adding via ObjectInstance arguments
        else:
            obj = self._ObjectClass(*data)
            self[obj.name] = obj

    def remove(self, name):
        if type(name) != str:
            raise TypeError("Argument must be the name of an object [str] to remove.")
        self.Dict.pop(name)

    # def add_property_to_all(self, property_name, item):
    #     for obj in self.values:
    #         setattr(obj, property_name, item)

    def get_index(self, idx):
        """
        Get an element of the object container ordered dict by referencing its index
        """
        return list(self.values)[idx]

    # ==============================================================================
    # ObjectContainer - Internal methods
    # ------------------------------------------------------------------------------
    # Mostly methods called by Model.initialize()
    # ==============================================================================

    # ==============================================================================
    # ObjectContainer - Printing/data-formatting related methods
    # ==============================================================================

    def get_pandas_dataframe(self, properties_to_print=None, include_idx=True):
        df = pandas.DataFrame()
        if include_idx:
            if properties_to_print and "idx" not in properties_to_print:
                properties_to_print.insert(0, "idx")
            for idx, (name, instance) in enumerate(self.items):
                df = pandas.concat(
                    [
                        df,
                        instance.get_pandas_series(properties_to_print=properties_to_print, idx=idx)
                        .to_frame()
                        .T,
                    ]
                )
        else:
            for idx, (name, instance) in enumerate(self.items):
                df = pandas.concat(
                    [
                        df,
                        instance.get_pandas_series(properties_to_print=properties_to_print)
                        .to_frame()
                        .T,
                    ]
                )
        # # sometimes types are recast. change entries into their original types
        # for dtypeName, dtype in self.dtypes.items():
        #     if dtypeName in df.columns:
        #         df = df.astype({dtypeName: dtype})

        return df

    def print_to_latex(
        self,
        properties_to_print=None,
        max_col_width=None,
        sig_figs=2,
        latex_name_map=None,
        return_df=True,
    ):
        """Requires latex packages \siunitx and \longtable"""

        df = self.get_pandas_dataframe_formatted(
            properties_to_print=properties_to_print,
            max_col_width=max_col_width,
            sig_figs=sig_figs,
        )

        # Change certain df entries to best format for
        for col in df.columns:
            # Convert quantity objects to unit
            if isinstance(df[col][0], pint.Quantity):
                # if tablefmt=='latex':
                df[col] = df[col].apply(lambda x: f"${x:0.{sig_figs}e~Lx}$")

            if col == "idx":
                df = df.drop("idx", axis=1)

        if return_df:
            return df
        else:
            with pandas.option_context("max_colwidth", 1000):
                logger.info(df.to_latex(escape=False, longtable=True, index=False))

    def get_pandas_dataframe_formatted(
        self, properties_to_print=None, max_col_width=50, sig_figs=2
    ):
        # Get the pandas dataframe with the properties we want to print
        if properties_to_print:
            if type(properties_to_print) != list:
                properties_to_print = [properties_to_print]
        elif hasattr(self, "properties_to_print"):
            properties_to_print = self.properties_to_print
        df = self.get_pandas_dataframe(properties_to_print=properties_to_print, include_idx=False)
        if properties_to_print:
            df = df[properties_to_print]

        # add new lines to df entries (type str) that exceed max col width
        if max_col_width:
            for col in df.columns:
                if isinstance(df[col][0], str):
                    df[col] = df[col].apply(lambda x: "\n".join(wrap(x, max_col_width)))

        # remove leading underscores from df column names (used for cached properties)
        for col in df.columns:
            # if isinstance(df[col][0], str) and col[0] == '_':
            if col[0] == "_":
                df.rename(columns={col: col[1:]}, inplace=True)

        return df

    def print(
        self,
        tablefmt="fancy_grid",
        properties_to_print=None,
        filename=None,
        max_col_width=50,
        sig_figs=2,
    ):
        df = self.get_pandas_dataframe_formatted(
            properties_to_print=properties_to_print,
            max_col_width=max_col_width,
            sig_figs=sig_figs,
        )

        # # Change certain df entries to best format for printing
        for col in df.columns:
            # Convert quantity objects to unit
            if isinstance(df[col][0], pint.Quantity):
                # if tablefmt=='latex':
                df[col] = df[col].apply(lambda x: f"{x:0.{sig_figs}e~P}")

        # print to file
        if filename is None:
            logger.info(tabulate(df, headers="keys", tablefmt=tablefmt))
        else:
            original_stdout = sys.stdout  # Save a reference to the original standard output
            with open(filename, "w") as f:  # TODO: Add this to logging
                # Change the standard output to the file we created.
                sys.stdout = f
                print("This message will be written to a file.")
                print(tabulate(df, headers="keys", tablefmt=tablefmt))  # ,
                sys.stdout = original_stdout  # Reset the standard output to its original value

    def __str__(self):
        df = self.get_pandas_dataframe(properties_to_print=self.properties_to_print)
        df = df[self.properties_to_print]

        return tabulate(df, headers="keys", tablefmt="fancy_grid")


class ObjectInstance:
    """
    Parent class containing general methods used by all stubs
    "objects": i.e. parameters, species, compartments, reactions, fluxes, forms
    """

    def _convert_pint_quantity_to_unit(self):
        # strip the magnitude and keep the units. Warn if magnitude!=1
        for name, attr in vars(self).items():
            if isinstance(attr, pint.Quantity):
                setattr(self, name, pint_quantity_to_unit(attr))

    def _check_input_type_validity(self):
        "Check that the inputs have the same type (or are convertible) to the type hint."
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.type == Any:
                continue
            elif not isinstance(value, field.type):
                try:
                    setattr(self, field.name, field.type(value))
                except Exception:
                    raise TypeError(
                        f"Object {self.name!r} type error: the attribute {field.name!r} "
                        f"is expected to be {field.type}, got {type(value)} instead. "
                        "Conversion to the expected type was attempted but unsuccessful."
                    )

    def _convert_pint_unit_to_quantity(self):
        # convert pint units to quantity
        for name, attr in vars(self).items():
            if isinstance(attr, pint.Unit):
                setattr(self, name, pint_unit_to_quantity(attr))

    def get_pandas_series(self, properties_to_print=None, idx=None):
        if properties_to_print:
            dict_to_convert = odict({"idx": idx})
            dict_to_convert.update(
                odict(
                    [
                        (key, val)
                        for (key, val) in self.__dict__.items()
                        if key in properties_to_print
                    ]
                )
            )
        else:
            dict_to_convert = self.__dict__
        return pandas.Series(dict_to_convert, name=self.name)
        # return pandas.Series(dict_to_convert)

    def print(self, properties_to_print=None):
        if rank == root:
            logger.info("Name: " + self.name)
            # if a custom list of properties to print is provided, only use those
            if properties_to_print:
                dict_to_print = dict(
                    [
                        (key, val)
                        for (key, val) in self.__dict__.items()
                        if key in properties_to_print
                    ]
                )
            else:
                dict_to_print = self.__dict__
                logger.info(pformat(dict_to_print, width=240))


# ==============================================================================
# ==============================================================================
# Classes for parameters, species, compartments, reactions, and fluxes
# ==============================================================================
# ==============================================================================


class ParameterContainer(ObjectContainer):
    def __init__(self):
        super().__init__(Parameter)

        self.properties_to_print = [
            "_quantity",
            "is_time_dependent",
            "sym_expr",
            "notes",
            "group",
        ]

    def print(
        self,
        tablefmt="fancy_grid",
        properties_to_print=None,
        filename=None,
        max_col_width=50,
    ):
        for s in self:
            s.quantity
        super().print(tablefmt, self.properties_to_print, filename, max_col_width)


@dataclass
class Parameter(ObjectInstance):
    name: str
    value: float
    unit: pint.Unit
    group: str = ""
    notes: str = ""
    use_preintegration: bool = False

    def to_dict(self):
        "Convert to a dict that can be used to recreate the object."
        keys_to_keep = [
            "name",
            "value",
            "unit",
            "group",
            "notes",
            "use_preintegration",
            "is_time_dependent",
            "is_space_dependent",
            "sym_expr",
            "preint_sym_expr",
            "sampling_data",
            "preint_sampling_data",
            "type",
            "free_symbols",
        ]
        return {key: self.__dict__[key] for key in keys_to_keep}

    @classmethod
    def from_dict(cls, input_dict):
        parameter = cls(input_dict["name"], input_dict["value"], input_dict["unit"])
        for key, val in input_dict.items():
            setattr(parameter, key, val)
        parameter.__post_init__()
        return parameter

    @classmethod
    def from_expression(
        cls,
        name,
        sym_expr,
        unit,
        preint_sym_expr=None,
        group="",
        notes="",
        use_preintegration=False,
    ):
        # Parse the given string to create a sympy expression
        if isinstance(sym_expr, str):
            sym_expr = parse_expr(sym_expr)
        x, y, z = (Symbol(f"x[{i}]") for i in range(3))
        sym_expr = sym_expr.subs({"x": x, "y": y, "z": z})

        # Check if expression is time/space dependent
        free_symbols = [str(x) for x in sym_expr.free_symbols]
        is_time_dependent = "t" in free_symbols
        is_space_dependent = not {"x[0]", "x[1]", "x[2]"}.isdisjoint(set(free_symbols))
        if is_space_dependent:
            raise NotImplementedError
        # For now, parameters can only be defined in terms of time/space
        if not {"x[0]", "x[1]", "x[2]", "t"}.issuperset(free_symbols):
            raise NotImplementedError

        # TODO: fix this when implementing space dependent parameters
        if is_time_dependent:
            value = float(sym_expr.subs({"t": 0.0}))

        parameter = cls(
            name,
            value,
            unit,
            group=group,
            notes=notes,
            use_preintegration=use_preintegration,
        )

        if use_preintegration:
            if preint_sym_expr:
                if isinstance(preint_sym_expr, str):
                    preint_sym_expr = parse_expr(preint_sym_expr)
                preint_sym_expr = preint_sym_expr.subs({"x": x, "y": y, "z": z})
            else:
                # try to integrate
                t = Symbol("t")
                preint_sym_expr = integrate(sym_expr, t)
            parameter.preint_sym_expr = preint_sym_expr

        parameter.free_symbols = free_symbols
        parameter.sym_expr = sym_expr
        parameter.is_time_dependent = is_time_dependent
        parameter.is_space_dependent = is_space_dependent

        # parameter.dolfin_expression = d.Expression(sym.printing.ccode(sym_expr), t=0.0, degree=1)
        parameter.type = "expression"
        parameter.__post_init__()
        logger.debug(
            f"Time-dependent parameter {name} evaluated from expression.",
            extra=dict(format_type="log"),
        )

        return parameter

    def __post_init__(self):
        if not hasattr(self, "is_time_dependent"):
            self.is_time_dependent = False
        if not hasattr(self, "is_space_dependent"):
            self.is_space_dependent = False

        if self.use_preintegration:
            logger.warning(
                f"Warning! Pre-integrating parameter {self.name}. Make sure that "
                f"expressions {self.name} appears in have no other time-dependent variables.",
                extra=dict(format_type="warning"),
            )

        attributes = [
            "sym_expr",
            "preint_sym_expr",
            "sampling_file",
            "sampling_data",
            "preint_sampling_data",
            "free_symbols",
        ]
        for attribute in attributes:
            if not hasattr(self, attribute):
                setattr(self, attribute, None)

        if not hasattr(self, "type"):
            self.type = "constant"

        self._convert_pint_quantity_to_unit()
        self._check_input_type_validity()
        self._convert_pint_unit_to_quantity()
        self.check_validity()

        # self.dolfin_constant = d.Constant(self.value)
        # self.value_unit = self.value*self.unit
        self.value_vector = np.array([0, self.value])

    @property
    def dolfin_quantity(self):
        if hasattr(self, "dolfin_expression") and not self.use_preintegration:
            return self.dolfin_expression * self.unit
        else:
            return self.dolfin_constant * self.unit

    @property
    def quantity(self):
        self._quantity = self.value * self.unit
        return self._quantity

    def check_validity(self):
        if self.is_time_dependent:
            if all(
                [x in ("", None) for x in [self.sampling_file, self.sym_expr, self.preint_sym_expr]]
            ):
                raise ValueError(
                    f"Parameter {self.name} is marked as time dependent "
                    "but is not defined in terms of time."
                )


class SpeciesContainer(ObjectContainer):
    def __init__(self):
        super().__init__(Species)
        self.properties_to_print = ["compartment_name", "dof_index", "_Diffusion"]

    def print(
        self,
        tablefmt="fancy_grid",
        properties_to_print=None,
        filename=None,
        max_col_width=50,
    ):
        for s in self:
            s.D_quantity
            s.latex_name
        super().print(tablefmt, self.properties_to_print, filename, max_col_width)

    def print_to_latex(
        self,
        properties_to_print=None,
        max_col_width=None,
        sig_figs=2,
        latex_name_map=None,
        return_df=False,
    ):
        properties_to_print = ["_latex_name"]
        properties_to_print.extend(self.properties_to_print)
        df = super().print_to_latex(
            properties_to_print, max_col_width, sig_figs, latex_name_map, return_df=True
        )
        # fix dof_index
        for col in df.columns:
            if col == "dof_index":
                df[col] = df[col].astype(int)
        # fix name
        # get the column of df that contains the name
        # this can be more robust
        # df.columns

        if return_df:
            return df
        else:
            with pandas.option_context("max_colwidth", 1000):
                logger.info(df.to_latex(escape=False, longtable=True, index=False))


@dataclass
class Species(ObjectInstance):
    name: str
    initial_condition: Any
    # initial_condition: float
    concentration_units: pint.Unit
    D: float
    diffusion_units: pint.Unit
    compartment_name: str
    group: str = ""

    def to_dict(self):
        "Convert to a dict that can be used to recreate the object."
        keys_to_keep = [
            "name",
            "initial_condition",
            "concentration_units",
            "D",
            "diffusion_units",
            "compartment_name",
            "group",
        ]
        return {key: self.__dict__[key] for key in keys_to_keep}

    @classmethod
    def from_dict(cls, input_dict):
        return cls(**input_dict)

    def __post_init__(self):
        # self.sub_species = {} # additional compartments this species
        # may live in in addition to its primary one
        self.is_in_a_reaction = False
        self.is_an_added_species = False
        self.dof_map = None
        self.u = dict()
        self._usplit = dict()
        self.ut = None
        self.v = None

        if isinstance(self.initial_condition, float):
            pass
        elif isinstance(self.initial_condition, int):
            self.initial_condition = float(self.initial_condition)
        elif isinstance(self.initial_condition, str):
            x, y, z = (Symbol(f"x[{i}]") for i in range(3))
            # Parse the given string to create a sympy expression
            sym_expr = parse_expr(self.initial_condition).subs({"x": x, "y": y, "z": z})

            # Check if expression is space dependent
            free_symbols = [str(x) for x in sym_expr.free_symbols]
            if not {"x[0]", "x[1]", "x[2]"}.issuperset(free_symbols):
                raise NotImplementedError
            logger.debug(
                f"Creating dolfin object for space-dependent initial condition {self.name}",
                extra=dict(format_type="log"),
            )
            self.initial_condition_expression = d.Expression(sym.printing.ccode(sym_expr), degree=1)
        else:
            raise TypeError("initial_condition must be a float or string.")

        self._convert_pint_quantity_to_unit()
        self._check_input_type_validity()
        self._convert_pint_unit_to_quantity()
        self.check_validity()

    def check_validity(self):
        # checking values
        if isinstance(self.initial_condition, float) and self.initial_condition < 0.0:
            raise ValueError(
                f"Initial condition for species {self.name} must be greater or equal to 0."
            )
        if self.D < 0.0:
            raise ValueError(
                f"Diffusion coefficient for species {self.name} must be greater or equal to 0."
            )
        # checking units
        if not self.diffusion_units.check("[length]^2/[time]"):
            raise ValueError(
                f"Units of diffusion coefficient for species {self.name} must "
                "be dimensionally equivalent to [length]^2/[time]."
            )

    @cached_property
    def vscalar(self):
        return d.TestFunction(sub(self.compartment.V, 0, True))

    @property
    def dolfin_quantity(self):
        return self._usplit["u"] * self.concentration_units

    @property
    def initial_condition_quantity(self):
        self._Initial_Concentration = self.initial_condition * self.concentration_units
        return self._Initial_Concentration

    @property
    def D_quantity(self):
        self._Diffusion = self.D * self.diffusion_units
        return self._Diffusion

    @property
    def sym(self):
        self._sym = Symbol(self.name)
        return self._sym

    @property
    def latex_name(self):
        # Change _ to - in name
        name = self.name.replace("_", "-")
        self._latex_name = sym.latex(Symbol(name))
        return self._latex_name


class CompartmentContainer(ObjectContainer):
    def __init__(self):
        super().__init__(Compartment)

        self.properties_to_print = [
            "_mesh_id",
            "dimensionality",
            "num_species",
            "_num_vertices",
            "_num_dofs",
            "_num_dofs_local",
            "_num_cells",
            "cell_marker",
            "_nvolume",
        ]

    def print(
        self,
        tablefmt="fancy_grid",
        properties_to_print=None,
        filename=None,
        max_col_width=50,
    ):
        for c in self:
            c.mesh_id
            c.nvolume
            c.num_vertices
            c.num_dofs
            c.num_dofs_local
            c.num_cells
        super().print(tablefmt, self.properties_to_print, filename, max_col_width)


@dataclass
class Compartment(ObjectInstance):
    name: str
    dimensionality: int
    compartment_units: pint.Unit
    cell_marker: Any

    def to_dict(self):
        "Convert to a dict that can be used to recreate the object."
        keys_to_keep = ["name", "dimensionality", "compartment_units", "cell_marker"]
        return {key: self.__dict__[key] for key in keys_to_keep}

    @classmethod
    def from_dict(cls, input_dict):
        return cls(**input_dict)

    def __post_init__(self):
        if (
            isinstance(self.cell_marker, list)
            and not all([isinstance(m, int) for m in self.cell_marker])
            or not isinstance(self.cell_marker, (int, list))
        ):
            raise TypeError("cell_marker must be an int or list of ints.")

        self._convert_pint_quantity_to_unit()
        self._check_input_type_validity()
        self._convert_pint_unit_to_quantity()
        self.check_validity()

        # Initialize
        self.species = odict()
        self.u = dict()
        self._usplit = dict()
        self.V = None
        self.v = None

    def check_validity(self):
        if self.dimensionality not in [1, 2, 3]:
            raise ValueError(
                f"Compartment {self.name} has dimensionality {self.dimensionality}. "
                "Dimensionality must be in [1,2,3]."
            )
        # checking units
        if not self.compartment_units.check("[length]"):
            raise ValueError(
                f"Compartment {self.name} has units of {self.compartment_units} "
                "- units must be dimensionally equivalent to [length]."
            )

    def specify_nonadjacency(self, nonadjacent_compartment_list=None):
        """
        Specify if this compartment is NOT adjacent to another compartment.
        Not necessary, but will speed-up initialization of very large problems.
        Only needs to be specified for surface meshes as those are the
        ones that MeshViews are built on.
        """
        if nonadjacent_compartment_list is None:
            self.nonadjacent_compartment_list = []
        self.nonadjacent_compartment_list = nonadjacent_compartment_list

    @property
    def measure_units(self):
        return self.compartment_units**self.dimensionality

    @property
    def mesh_id(self):
        self._mesh_id = self.mesh.id
        return self._mesh_id

    @property
    def dolfin_mesh(self):
        return self.mesh.dolfin_mesh

    @property
    def nvolume(self):
        "nvolume with proper units"
        self._nvolume = self.mesh.nvolume * self.compartment_units**self.dimensionality
        return self._nvolume

    @property
    def num_cells(self):
        self._num_cells = self.mesh.num_cells
        return self._num_cells

    @property
    def num_facets(self):
        self._num_facets = self.mesh.num_facets
        return self._num_facets

    @property
    def num_vertices(self):
        self._num_vertices = self.mesh.num_vertices
        return self._num_vertices

    @property
    def num_dofs(self):
        "Number of degrees of freedom for this compartment"
        # self._num_dofs = self.num_species * self.num_vertices
        # return self._num_dofs
        if self.V is None:
            self._num_dofs = 0
        else:
            self._num_dofs = self.V.dim()
        return self._num_dofs

    @property
    def num_dofs_local(self):
        "Number of degrees of freedom for this compartment, local to this process"
        if self.V is None:
            self._num_dofs_local = 0
        else:
            self._ownership_range = self.V.dofmap().ownership_range()
            self._num_dofs_local = self._ownership_range[1] - self._ownership_range[0]
        return self._num_dofs_local


class ReactionContainer(ObjectContainer):
    def __init__(self):
        super().__init__(Reaction)

        self.properties_to_print = ["lhs", "rhs", "eqn_f_str", "eqn_r_str"]

    def print(
        self,
        tablefmt="fancy_grid",
        properties_to_print=None,
        filename=None,
        max_col_width=50,
    ):
        super().print(tablefmt, self.properties_to_print, filename, max_col_width)


@dataclass
class Reaction(ObjectInstance):
    name: str
    lhs: list
    rhs: list
    param_map: dict
    species_map: dict = dataclasses.field(default_factory=dict)
    flux_scaling: dict = dataclasses.field(default_factory=dict)
    reaction_type: str = "mass_action"
    explicit_restriction_to_domain: str = ""
    track_value: bool = False
    eqn_f_str: str = ""
    eqn_r_str: str = ""
    group: str = ""

    def to_dict(self):
        "Convert to a dict that can be used to recreate the object."
        keys_to_keep = [
            "name",
            "lhs",
            "rhs",
            "param_map",
            "species_map",
            "flux_scaling",
            "reaction_type",
            "explicit_restriction_to_domain",
            "track_value",
            "eqn_f_str",
            "eqn_r_str",
            "group",
        ]
        return {key: self.__dict__[key] for key in keys_to_keep}

    @classmethod
    def from_dict(cls, input_dict):
        return cls(**input_dict)

    def __post_init__(self):
        self._check_input_type_validity()
        self.check_validity()
        self.fluxes = dict()

        if self.eqn_f_str != "" or self.eqn_r_str != "" and self.reaction_type == "mass_action":
            self.reaction_type = "custom"

        # Finish initializing the species map
        for species_name in set(self.lhs + self.rhs):
            if species_name not in self.species_map:
                self.species_map[species_name] = species_name

        # Finish initializing the species flux scaling
        for species_name in set(self.lhs + self.rhs):
            if species_name not in self.flux_scaling:
                self.flux_scaling[species_name] = None

    def check_validity(self):
        # Type checking
        if not all([isinstance(x, str) for x in self.lhs]):
            raise TypeError(f"Reaction {self.name} requires a list of strings as input for lhs.")
        if not all([isinstance(x, str) for x in self.rhs]):
            raise TypeError(f"Reaction {self.name} requires a list of strings as input for rhs.")
        if not all([type(k) == str and type(v) == str for (k, v) in self.param_map.items()]):
            raise TypeError(
                f"Reaction {self.name} requires a dict of str:str as input for param_map."
            )
        if self.species_map:
            if not all(
                [isinstance(k, str) and isinstance(v, str) for (k, v) in self.species_map.items()]
            ):
                raise TypeError(
                    f"Reaction {self.name} requires a dict of str:str as input for species_map."
                )
        if self.flux_scaling:
            if not all(
                [
                    isinstance(k, str) and isinstance(v, (numbers.Number, None))
                    for (k, v) in self.flux_scaling.items()
                ]
            ):
                raise TypeError(
                    f"Reaction {self.name} requires a dict of "
                    "str:number as input for flux_scaling."
                )

    def _parse_custom_reaction(self, reaction_eqn_str):
        reaction_expr = parse_expr(reaction_eqn_str)
        reaction_expr = reaction_expr.subs(self.param_map)
        # # use species_scaling if available
        # reaction_expr = reaction_expr.subs(self._species_scaling_map)
        reaction_expr = reaction_expr.subs(self.species_map)
        return str(reaction_expr)

    def reaction_to_fluxes(self):
        logger.debug(f"Getting fluxes for reaction {self.name}", extra=dict(format_type="log"))
        # set of 2-tuples. (species_name, signed stoichiometry)
        self.species_stoich = {
            (species_name, -1 * self.lhs.count(species_name)) for species_name in self.lhs
        }
        self.species_stoich.update(
            {(species_name, 1 * self.rhs.count(species_name)) for species_name in self.rhs}
        )
        # convert to dict
        self.species_stoich = dict(self.species_stoich)
        # combine with user-defined flux scaling
        for species_name in self.species_stoich.keys():
            if self.flux_scaling[species_name] is not None:
                logger.debug(
                    f"Flux {self.name}: stoichiometry/flux for species {species_name} "
                    f"scaled by {self.flux_scaling[species_name]}",
                    extra=dict(format_type="log"),
                )
                self.species_stoich[species_name] *= self.flux_scaling[species_name]

        for species_name, stoich in self.species_stoich.items():
            species = self.species[species_name]
            if self.eqn_f_str:
                flux_name = self.name + f" [{species_name} (f)]"
                eqn = stoich * parse_expr(self.eqn_f_str)
                self.fluxes.update({flux_name: Flux(flux_name, species, eqn, self)})
            if self.eqn_r_str:
                flux_name = self.name + f" [{species_name} (r)]"
                eqn = -stoich * parse_expr(self.eqn_r_str)
                self.fluxes.update({flux_name: Flux(flux_name, species, eqn, self)})


class FluxContainer(ObjectContainer):
    def __init__(self):
        super().__init__(Flux)

        self.properties_to_print = [
            "_species_name",
            "equation",
            "topology",
            "_assembled_flux",
        ]

    def print(
        self,
        tablefmt="fancy_grid",
        properties_to_print=None,
        filename=None,
        max_col_width=50,
    ):
        for f in self:
            f.assembled_flux
            f.equation_lambda_eval("quantity")
        super().print(tablefmt, self.properties_to_print, filename, max_col_width)


@dataclass
class Flux(ObjectInstance):
    name: str
    destination_species: Species
    equation: sym.Expr
    reaction: Reaction

    def check_validity(self):
        pass

    def __post_init__(self):
        # for nice printing
        self._species_name = self.destination_species.name

        self._check_input_type_validity()
        self.check_validity()

        # Add in an uninitialized unit_scale_factor
        self.unit_scale_factor = 1.0 * unit.dimensionless
        self.equation = self.equation * Symbol("unit_scale_factor")

        # Getting additional flux properties
        self._post_init_get_involved_species_parameters_compartments()
        self._post_init_get_flux_topology()

        # Get equation lambda expression
        self.equation_lambda = sym.lambdify(
            list(self.equation_variables.keys()),
            self.equation,
            modules=common.stubs_expressions(gset["dolfin_expressions"]),
        )

        # Update equation with correct unit scale factor
        self._post_init_get_flux_units()
        self._post_init_get_integration_measure()

        # Check if flux is linear with respect to different components
        self.is_linear_wrt_comp = dict()
        self._post_init_get_is_linear_comp()

    def _post_init_get_involved_species_parameters_compartments(self):
        self.destination_compartment = self.destination_species.compartment

        # Get the subset of species/parameters/compartments that are relevant
        variables = {str(x) for x in self.equation.free_symbols}
        all_params = self.reaction.parameters
        all_species = self.reaction.species
        self.parameters = {x: all_params[x] for x in variables.intersection(all_params.keys())}
        self.species = {x: all_species[x] for x in variables.intersection(all_species.keys())}
        self.compartments = self.reaction.compartments

    def _post_init_get_flux_topology(self):
        """
        Previous flux topology types:
        [1d] volume:                    PDE of u
        [1d] surface:                   PDE of v
        [2d] volume_to_surface:         PDE of v
        [2d] surface_to_volume:         BC of u

        Flux topology types:
        [1d] volume:                    PDE of u
        [1d] surface:                   PDE of v
        [2d] volume_to_surface:         PDE of v
        [2d] surface_to_volume:         BC of u
        [3d] volume-surface_to_volume:  BC of u ()
        [3d] volume-volume_to_surface:  PDE of v ()
        """
        # 1 compartment flux
        if self.reaction.topology in ["surface", "volume"]:
            self.topology = self.reaction.topology
            source_compartments = {self.destination_compartment.name}
        # 2 or 3 compartment flux
        elif self.reaction.topology in ["volume_surface", "volume_surface_volume"]:
            source_compartments = set(self.compartments.keys()).difference(
                {self.destination_compartment.name}
            )

            if self.reaction.topology == "volume_surface":
                assert len(source_compartments) == 1
                if self.destination_compartment.is_volume:
                    self.topology = "surface_to_volume"
                else:
                    self.topology = "volume_to_surface"

            elif self.reaction.topology == "volume_surface_volume":
                assert len(source_compartments) == 2
                if self.destination_compartment.is_volume:
                    self.topology = "volume-surface_to_volume"
                else:
                    self.topology = "volume-volume_to_surface"

        else:
            raise AssertionError()

        self.source_compartments = {
            name: self.reaction.compartments[name] for name in source_compartments
        }
        self.surface = [c for c in self.compartments.values() if c.mesh.is_surface]
        if len(self.surface) == 1:
            self.surface = self.surface[0]
        else:
            self.surface = None

        # Based on topology we know if it is a boundary condition or RHS term
        if self.topology in [
            "volume",
            "surface",
            "volume_to_surface",
            "volume-volume_to_surface",
        ]:
            self.is_boundary_condition = False
        elif self.topology in ["surface_to_volume", "volume-surface_to_volume"]:
            self.is_boundary_condition = True
        else:
            raise AssertionError()

    def _post_init_get_flux_units(self):
        concentration_units = self.destination_species.concentration_units
        compartment_units = self.destination_compartment.compartment_units
        diffusion_units = self.destination_species.diffusion_units

        # The expected units
        if self.is_boundary_condition:
            self._expected_flux_units = (
                1.0 * concentration_units / compartment_units * diffusion_units
            )  # ~D*du/dn
        else:
            self._expected_flux_units = 1.0 * concentration_units / unit.s  # rhs term. ~du/dt

        # Use the uninitialized unit_scale_factor to get the actual units
        # this is redundant if called by __post_init__
        self.unit_scale_factor = 1.0 * unit.dimensionless
        initial_equation_units = self.equation_lambda_eval("units")

        # If unit dimensionality is not correct a parameter likely needs to be adjusted
        if self._expected_flux_units.dimensionality != initial_equation_units.dimensionality:
            logger.info(self.unit_scale_factor)
            raise ValueError(
                f"Flux {self.name} has wrong units (cannot be converted) "
                f"- expected {self._expected_flux_units}, got {initial_equation_units}."
            )
        # Fix scaling
        else:
            # Define new unit_scale_factor, and update equation_units
            # by re-evaluating the lambda expression
            self.unit_scale_factor = (
                initial_equation_units.to(self._expected_flux_units) / initial_equation_units
            )
            self.equation_units = self.equation_lambda_eval(
                "units"
            )  # these should now be the proper units

            # should be redundant with previous checks, but just in case
            assert self.unit_scale_factor.dimensionless
            assert (
                initial_equation_units * self.unit_scale_factor
            ).units == self._expected_flux_units
            assert self.equation_units == self._expected_flux_units

            # If we already have the correct units, there is no need to update the equation
            if self.unit_scale_factor.magnitude == 1.0:
                return

            logger.debug(
                f"Flux {self.name} scaled by {self.unit_scale_factor}",
                extra=dict(new_lines=[1, 0], format_type="log"),
            )
            logger.debug(f"Old flux units: {self.equation_units}", extra=dict(format_type="log"))
            logger.debug(
                f"New flux units: {self._expected_flux_units}",
                extra=dict(new_lines=[0, 1], format_type="log"),
            )
            print("")

    def _post_init_get_integration_measure(self):
        """
        Flux topologys:
        [1d] volume:                    PDE of u
        [1d] surface:                   PDE of v
        [2d] volume_to_surface:         PDE of v
        [2d] surface_to_volume:         BC of u
        [3d] volume-surface_to_volume:  BC of u ()
        [3d] volume-volume_to_surface:  PDE of v ()

        1d means no other compartment are involved, so the integration
        measure is the volume of the compartment
        2d/3d means two/three compartments are involved, so the
        integration measure is the intersection between all compartments
        """

        if self.topology in ["volume", "surface"]:
            self.measure = self.destination_compartment.mesh.dx
            self.measure_units = self.destination_compartment.measure_units
        elif self.topology in [
            "volume_to_surface",
            "surface_to_volume",
            "volume-volume_to_surface",
            "volume-surface_to_volume",
        ]:
            # intersection of this surface with boundary of source volume(s)
            logger.debug(
                "DEBUGGING INTEGRATION MEASURE (only fully defined domains are enabled for now)"
            )
            self.measure = self.surface.mesh.dx
            self.measure_units = self.surface.compartment_units**self.surface.dimensionality

    # We define this as a property so that it is automatically updated

    @property
    def equation_variables(self):
        variables = {
            variable.name: variable.dolfin_quantity
            for variable in {**self.parameters, **self.species}.values()
        }
        variables.update({"unit_scale_factor": self.unit_scale_factor})
        return variables

    # @property
    # def equation_value(self):
    #     return self.equation_quantity.magnitude
    # @property
    # def equation_units(self):
    #     return common.pint_unit_to_quantity(self.equation_quantity.units)

    def equation_lambda_eval(self, input_type="quantity"):
        """
        Evaluates the equation lambda function using either the quantity
        (value * units), the value, or the units.
        The values and units are evaluated separately and then combined
        because some expressions don't work well
        with pint quantity types.
        """
        # This is an attempt to make the equation lambda work with pint quantities
        self._equation_quantity = self.equation_lambda(**self.equation_variables)
        if input_type == "quantity":
            return self._equation_quantity
        elif input_type == "value":
            return self._equation_quantity.magnitude
        elif input_type == "units":
            return common.pint_unit_to_quantity(self._equation_quantity.units)

    # Seems like setting this as a @property doesn't cause fenics to recompile

    @property
    def form(self):
        """-1 factor because terms are defined as if they were on the
        lhs of the equation F(u;v)=0"""
        return (
            d.Constant(-1)
            * self.equation_lambda_eval(input_type="value")
            * self.destination_species.v
            * self.measure
        )

    @property
    def scalar_form(self):
        """if the destination species is a vector function,
        the assembled form will be a vector of size NDOF."""
        return (
            d.Constant(-1)
            * self.equation_lambda_eval(input_type="value")
            * self.destination_species.vscalar
            * self.measure
        )

    @property
    def assembled_flux(self):
        """Same thing as molecules_per_second but doesn't try to convert
        units (e.g. volumetric concentration is being used on a 2d domain)"""
        try:
            self._assembled_flux = -1 * (
                d.assemble(self.scalar_form).sum() * self.equation_units * self.measure_units
            ).to(unit.molecule / unit.s)
        except Exception:
            self._assembled_flux = -1 * (
                d.assemble(self.scalar_form).sum() * self.equation_units * self.measure_units
            )
        return self._assembled_flux

    def _post_init_get_is_linear_comp(self):
        """
        Is the flux linear in terms of a compartment vector (e.g. dj/du['pm'])
        """
        umap = {}

        for species_name, species in self.species.items():
            comp_name = species.compartment_name
            umap.update({species_name: "u" + comp_name})

        uset = set(umap.values())
        new_eqn = self.equation.subs(umap)

        for comp_name in self.compartments.keys():
            d_new_eqn = sym.diff(new_eqn, "u" + comp_name, 1)
            d_new_eqn_species = {str(x) for x in d_new_eqn.free_symbols}
            self.is_linear_wrt_comp[comp_name] = uset.isdisjoint(d_new_eqn_species)


class FormContainer(ObjectContainer):
    def __init__(self):
        super().__init__(Form)

        self.properties_to_print = ["form", "form_type", "_compartment_name"]

    def print(self, tablefmt="fancy_grid", properties_to_print=None):
        for f in self:
            f.integrals
        super().print(tablefmt, self.properties_to_print)


@dataclass
class Form(ObjectInstance):
    """
    form_type:
    'mass': transient/mass form (holds time derivative)
    'diffusion'
    'domain_reaction'
    'boundary_reaction'

    Differentiating using ufl doesn't seem to get it right when
    using vector functions. Luckily we have all fluxes as sympy objects
    and mass/diffusive forms are always linear w.r.t components.
    """

    name: str
    form_: ufl.Form
    species: Species
    form_type: str
    units: pint.Unit
    is_lhs: bool
    linear_wrt_comp: dict = dataclasses.field(default_factory=dict)
    form_scaling: float = 1.0

    def set_scaling(self, form_scaling=1.0, print_scaling=True):
        self.form_scaling = form_scaling
        self.form_scaling_dolfin_constant.assign(self.form_scaling)
        if print_scaling:
            logger.info(
                f"Form scaling for form {self.name} set to {self.form_scaling}",
                extra=dict(format_type="log"),
            )

    @property
    def form(self):
        return self.form_scaling_dolfin_constant * self.form_

    @property
    def lhs(self):
        if self.is_lhs:
            return self.form
        else:
            return d.Constant(-1) * self.form

    @property
    def rhs(self):
        if self.is_lhs:
            return d.Constant(-1) * self.form
        else:
            return self.form

    def __post_init__(self):
        self.compartment = self.species.compartment
        self._compartment_name = self.compartment.name

        self.form_scaling_dolfin_constant = d.Constant(self.form_scaling)

        self._convert_pint_quantity_to_unit()
        self._check_input_type_validity()
        self._convert_pint_unit_to_quantity()

    @property
    def integrals(self):
        self._integrals = self.form.integrals()
        return self._integrals


def empty_sbmodel():
    pc = ParameterContainer()
    sc = SpeciesContainer()
    cc = CompartmentContainer()
    rc = ReactionContainer()
    return pc, sc, cc, rc


def sbmodel_from_locals(local_values):
    # FIXME: Add typing
    # Initialize containers
    pc, sc, cc, rc = empty_sbmodel()
    parameters = [x for x in local_values if isinstance(x, Parameter)]
    species = [x for x in local_values if isinstance(x, Species)]
    compartments = [x for x in local_values if isinstance(x, Compartment)]
    reactions = [x for x in local_values if isinstance(x, Reaction)]
    # we just reverse the list so that the order is the same as how they were defined
    parameters.reverse()
    species.reverse()
    compartments.reverse()
    reactions.reverse()
    pc.add(parameters)
    sc.add(species)
    cc.add(compartments)
    rc.add(reactions)
    return pc, sc, cc, rc


def nan_to_none(df):
    return df.replace({np.nan: None})
