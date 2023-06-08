"""Classes for parameters, species, compartments, reactions, fluxes, and forms.
Model class contains functions to efficiently solve a system.
"""
import dataclasses
import logging
import numbers
import sys
from collections import OrderedDict as odict
from dataclasses import dataclass
from enum import Enum
from pprint import pformat
from textwrap import wrap
from typing import Any, List, Optional, Union
import warnings

import dolfin as d
import numpy as np
import pandas
import pint
import sympy as sym
import ufl
from cached_property import cached_property
from sympy import Symbol, integrate
from sympy.parsing.sympy_parser import parse_expr
from tabulate import tabulate

from . import common
from .config import global_settings as gset
from .units import quantity_to_unit, unit, unit_to_quantity

__all__ = [
    "empty_sbmodel",
    "sbmodel_from_locals",
    "Compartment",
    "Parameter",
    "Reaction",
    "Species",
    "sbmodel_from_locals",
    "ParameterType",
]

comm = d.MPI.comm_world
rank = comm.rank
size = comm.size
root = 0

logger = logging.getLogger(__name__)


# ====================================================
# ====================================================
# Base Classes
# ====================================================
# ====================================================


class ParameterType(str, Enum):
    from_file = "from_file"
    constant = "constant"
    expression = "expression"


class InvalidObjectException(Exception):
    pass


class ObjectContainer:
    """Parent class containing general methods used by all "containers" """

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
        """Size of ObjectContainer"""
        return len(self.Dict)

    @property
    def items(self):
        """Return all items in container"""
        return self.Dict.items()

    @property
    def values(self):
        """Return all Dict values"""
        return self.Dict.values()

    def __iter__(self):
        """Iteratively return Dict values"""
        return iter(self.Dict.values())

    @property
    def keys(self):
        """Return all Dict keys"""
        return self.Dict.keys()

    @property
    def indices(self):
        return enumerate(self.keys)

    def to_dicts(self):
        """Returns a list of dicts that can be used to recreate the ObjectContainer"""
        return [obj.to_dict() for obj in self.values]

    def __getitem__(self, key):
        """syntactic sugar to allow: :code:`objcontainer[key] = objcontainer[key]`"""
        return self.Dict[key]

    def __setitem__(self, key, newvalue):
        """syntactic sugar to allow: :code:`objcontainer[key] = objcontainer[key]`"""
        self.Dict[key] = newvalue

    def add(self, *data):
        """Add data to object container"""
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
                        raise InvalidObjectException("Could not add data to ObjectContainer")
        # Adding via ObjectInstance arguments
        else:
            obj = self._ObjectClass(*data)
            self[obj.name] = obj

    def remove(self, name):
        """Remove data from object container"""
        if type(name) != str:
            raise TypeError("Argument must be the name of an object [str] to remove.")
        self.Dict.pop(name)

    def get_index(self, idx):
        """Get an element of the object container ordered dict by referencing its index"""
        return list(self.values)[idx]

    # ==============================================================================
    # ObjectContainer - Printing/data-formatting related methods
    # ==============================================================================

    def get_pandas_dataframe(
        self, properties_to_print: Optional[List[str]] = None, include_idx: bool = True
    ) -> pandas.DataFrame:
        """
        Create a `pandas.DataFrame` of all items in the class (defined through `self.items`)

        Args:
            properties_to_print: If set only the listed properties (by attribute name)
              is added to the series
            include_index: If true, add index as the first column in the data-frame.
        """

        df = pandas.DataFrame()
        if include_idx:
            if properties_to_print is not None and "idx" not in properties_to_print:
                properties_to_print.insert(0, "idx")
            for idx, (_, instance) in enumerate(self.items):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # See https://github.com/hgrecco/pint-pandas/issues/128
                    df = pandas.concat(
                        [
                            df,
                            instance.get_pandas_series(
                                properties_to_print=properties_to_print, idx=idx
                            )
                            .to_frame()
                            .T,
                        ]
                    )
        else:
            for idx, (_, instance) in enumerate(self.items):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # See https://github.com/hgrecco/pint-pandas/issues/128
                    df = pandas.concat(
                        [
                            df,
                            instance.get_pandas_series(properties_to_print=properties_to_print)
                            .to_frame()
                            .T,
                        ]
                    )

        return df

    def print_to_latex(
        self,
        properties_to_print=None,
        max_col_width=None,
        sig_figs=2,
        return_df=True,
    ):
        """
        Print object properties in latex format.
        Requires latex packages :code:`\siunitx` and :code:`\longtable`
        """

        df = self.get_pandas_dataframe_formatted(
            properties_to_print=properties_to_print,
            max_col_width=max_col_width,
            sig_figs=sig_figs,
        )

        # Change certain df entries to best format for display
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
        self,
        properties_to_print: Optional[Union[str, List[str]]] = None,
        max_col_width=50,
        sig_figs=2,
    ):
        """Get formatted pandas dataframe for printing object properties."""
        # Get the pandas dataframe with the properties we want to print
        if properties_to_print is not None:
            if not isinstance(properties_to_print, list):
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
        """Print object properties to file and/or terminal."""

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
            logger.info(
                tabulate(df, headers="keys", tablefmt=tablefmt),
                extra=dict(format_type="table"),
            )
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
    """Parent class containing general methods used by all smart
    "objects": i.e. parameters, species, compartments, reactions, fluxes, forms
    """

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

    def _convert_pint_quantity_to_unit(self):
        """Convert all attributes of the class that is a :code:`pint.Quantity`
        into a :code:`pint.Unit`."""
        # strip the magnitude and keep the units. Warn if magnitude!=1
        for name, attr in vars(self).items():
            if isinstance(attr, pint.Quantity):
                setattr(self, name, quantity_to_unit(attr))

    def _convert_pint_unit_to_quantity(self):
        """Convert all attributes of the class that is a :code:`pint.Unit`
        into a :code:`pint.Quantity`."""
        for name, attr in vars(self).items():
            if isinstance(attr, pint.Unit):
                setattr(self, name, unit_to_quantity(attr))

    def get_pandas_series(
        self, properties_to_print: Optional[List[str]] = None, idx: Optional[int] = None
    ):
        """
        Convert attributes of the class into a :code:`pandas.Series`.

        Args:
            properties_to_print: If set only the listed properties (by attribute name)
                is added to the series
            index: If set add to series
        """
        if properties_to_print is not None:
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

    def print(self, properties_to_print=None):
        """Print properties in current object instance."""
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
    """
    Parameter objects contain information for the various parameters involved in reactions,
    such as binding rates and dissociation constants.
    A Parameter object that is constant over time and space is initialized by calling

    .. code:: python

    param_var = Parameter(
        name, value, unit, group (opt), notes (opt),
        use_preintegration (opt)
        )

    Args:
        name: string naming the parameter
        value: value of the given parameter
        unit: units associated with given value
        group (optional): string placing this parameter in a designated group;
             for organizational purposes when there are multiple reaction modules
        notes (optional): string related to this parameter
        use_preintegration (optional): not applicable for constant parameter

    To initialize a parameter object that varies over time, you can either
    specify a string that gives the parameter as a function of time (t)
    or load data from a .txt file.

    To load from a string expression, call:

    .. code:: python

        param_var = Parameter.from_expression(
            name, sym_expr, unit, preint_sym_expr (opt), group (opt),
            notes (opt), use_preintegration (opt)
        )

    Inputs are the same as described above, except:

    Args:
        sym_expr: string specifying an expression, "t" should be the only free variable
        preint_sym_expr (optional): string giving the integral of the expression; if not given
                                  and use_preintegration is true, then sympy tries to integrate
                                  using sympy.integrate()
        use_preintegration (optional):  use preintegration in solution process if
                                     "use_preintegration" is true (defaults to false)

    To load parameter over time from a .txt file, call:

    .. code:: python

        param_var = Parameter.from_file(
            name, sampling_file, unit, group (opt),
            notes (opt), use_preintegration (opt)
        )

    Inputs are the same as described above, except:

    Args:
        sampling_file: name of text file giving parameter data in two columns (comma-separated) -
                     first column is time (starting with t=0.0) and second is parameter values
        use_preintegration (optional):  use preintegration in solution process if
                                     "use_preintegration" is true (defaults to false),
                                     uses sci.integrate.cumtrapz for numerical integration
    """

    name: str
    value: float
    unit: pint.Unit
    group: str = ""
    notes: str = ""
    use_preintegration: bool = False

    def to_dict(self):
        """Convert to a dict that can be used to recreate the object."""
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
        """Read parameter object from Dict"""
        parameter = cls(input_dict["name"], input_dict["value"], input_dict["unit"])
        for key, val in input_dict.items():
            setattr(parameter, key, val)
        parameter.__post_init__()
        return parameter

    @classmethod
    def from_file(cls, name, sampling_file, unit, group="", notes="", use_preintegration=False):
        """ "
        Load in a purely time-dependent scalar function from data
        Data needs to be read in from a text file with two columns
        where the first column is time (first entry must be 0.0)
        and the second column is the parameter values.
        Columns should be comma-separated.
        """
        # load in sampling data file
        sampling_data = np.genfromtxt(sampling_file, dtype="float", delimiter=",")
        logger.info(f"Loading in data for parameter {name}", extra=dict(format_type="log"))
        if sampling_data[0, 0] != 0.0 or sampling_data.shape[1] != 2:
            raise NotImplementedError
        value = sampling_data[0, 1]  # initial value

        parameter = cls(
            name,
            value,
            unit,
            group=group,
            notes=notes,
            use_preintegration=use_preintegration,
        )

        if use_preintegration:
            # preintegrate sampling data using cumtrapz
            from scipy.integrate import cumtrapz

            int_data = cumtrapz(sampling_data[:, 1], x=sampling_data[:, 0], initial=0)
            # concatenate time vector
            preint_sampling_data = np.hstack(
                sampling_data[:, 0].reshape(-1, 1), int_data.reshape(-1, 1)
            )
            parameter.preint_sampling_data = preint_sampling_data

        # initialize instance
        parameter.sampling_file = sampling_file
        parameter.sampling_data = sampling_data
        parameter.is_time_dependent = True
        parameter.is_space_dependent = False  # not supported yet
        parameter.type = ParameterType.from_file
        parameter.__post_init__()
        logger.info(
            f"Time-dependent parameter {name} loaded from file.",
            extra=dict(format_type="log"),
        )

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
        """Use sympy to parse time-dependent expression for parameter"""
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
        parameter.type = ParameterType.expression
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
            self.type = ParameterType.constant

        self._convert_pint_quantity_to_unit()
        self._check_input_type_validity()
        self._convert_pint_unit_to_quantity()
        self.check_validity()
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
        """Confirm that time-dependent parameter is defined in terms of time"""
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
        return_df=False,
    ):
        properties_to_print = ["_latex_name"]
        properties_to_print.extend(self.properties_to_print)
        df = super().print_to_latex(properties_to_print, max_col_width, sig_figs, return_df=True)
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
    """
    Each Species object contains information for one state variable in the model
    (can be a molecule, receptor open probability, membrane voltage, etc.)

    Args:
        name: string naming the species
        conc_init: initial concentration for this species
            (can be an expression given by a string to be parsed by sympy
            - the only unknowns in the expression should be x, y, and z)
        conc_units: concentration units for this species
        D: diffusion coefficient value
        diffusion_units: units for diffusion coefficient
        compartment_name: each species should be assigned to a single compartment
        group (optional): for larger models, specifies a group of species this belongs to;
            for organizational purposes when there are multiple reaction modules

    Species object is initialized by calling:

    .. code:: python

        species_var = Species(
            name, initial_condition, concentration_units,
            D, diffusion_units, compartment_name, group (opt)
        )
    """

    name: str
    initial_condition: Any
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
        """Load Species object from Dict"""
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
        """Species validity checks:

        * Initial condition is greater than or equal to 0
        * Diffusion coefficient is greater than or equal to 0
        * Diffusion coefficient has units of length^2/time
        """
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
        return d.TestFunction(common.sub(self.compartment.V, 0, True))

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
    """Each Compartment object contains information describing a surface, volume, or edge
    within the geometry of interest. The object is initialized by calling:

        .. code:: python

            compartment_var = Compartment(name, dimensionality, compartment_units, cell_marker)

    Args:
        name: string naming the compartment
        dimensionality: topological dimensionality (e.g. 3 for volume, 2 for surface)
        compartment_units: length units for the compartment
        cell_marker: marker value identifying the compartment in the parent mesh
    """

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
        """
        Compartment validity checks:

        * Compartment dimensionality is 1,2, or 3
        * Compartment units are of type "length"
        """
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
        """Specify if this compartment is NOT adjacent to another compartment.
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
        """nvolume with proper units"""
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
        """Number of degrees of freedom for this compartment"""
        # self._num_dofs = self.num_species * self.num_vertices
        # return self._num_dofs
        if self.V is None:
            self._num_dofs = 0
        else:
            self._num_dofs = self.V.dim()
        return self._num_dofs

    @property
    def num_dofs_local(self):
        """Number of degrees of freedom for this compartment, local to this process"""
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
    """
    A Reaction object contains information on a single biochemical interaction
    between species in a single compartment or across multiple compartments.

    Args:
        name: string naming the reaction
        lhs: list of strings specifying the reactants for this reaction
        rhs: list of strings specifying the products for this reaction
            NOTE: the lists "lhs" and "rhs" determine the stoichiometry of the reaction;
            for instance, if two A's react to give one B, the reactants list would be
            :code:`["A","A"]`, and the products list would be :code:`["B"]`
        param_map: relationship between the parameters specified in the reaction string
            and those given in the parameter container. By default, the reaction parameters are
            "on" and "off" when a system obeys simple mass action.
            If the forward rate is given by a parameter :code:`k1` and the reverse
            rate is given by :code:`k2`, then :code:`param_map = {"on":"k1", "off":"k2"}`
        eqn_f_str: For systems not obeying simple mass action,
            this string specifies the forward reaction rate By default,
            this string is "on*{all reactants multiplied together}"
        eqn_r_str: For systems not obeying simple mass action,
            this string specifies the reverse reaction rate
            By default, this string is :code:`off*{all products multiplied together}`
            reaction_type: either "custom" or "mass_action" (default is "mass_action")
            [never a required argument]
        species_map: same format as param_map;
            required if the species does not appear in the lhs or rhs lists
        explicit_restriction_to_domain: string specifying where the reaction occurs;
            required if the reaction is not constrained by the reaction string
            (e.g., if production occurs only at the boundary,
            but the species being produced exists through the entire volume)
        group: string placing this reaction in a reaction group;
            for organizational purposes when there are multiple reaction modules
            flux_scaling: in certain cases, a given reactant or product
            may experience a scaled flux (for instance, if we assume that
            some of the molecules are immediately sequestered after the reaction);
            in this case, to signify that this flux should be rescaled, we specify
            :code:`flux_scaling = {scaled_species: scale_factor}`,
            where scaled_species is a string specifying the species to be scaled and
            scale_factor is a number specifying the rescaling factor

    .. note::

        The Reaction object is initialized by calling:

        .. code:: python

            reaction_name = Reaction(
                name, lhs, rhs, param_map,
                eqn_f_str (opt), eqn_r_str (opt), reaction_type (opt), species_map,
                explicit_restriction_to_domain (opt), group (opt), flux_scaling (opt)
            )
    """

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
        """
        Reaction validity checks:

        * LHS (reactants) and RHS (products) are specified as lists of strings
        * param_map must be specified as a dict of "str:str"
        * Species_map must be specified as a dict of "str:str"
        * If given, flux scaling must be specified as a dict of "species:scale_factor"
        """
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
        "Substitute parameters and species into reaction expression"
        reaction_expr = parse_expr(reaction_eqn_str)
        reaction_expr = reaction_expr.subs(self.param_map)
        reaction_expr = reaction_expr.subs(self.species_map)
        return str(reaction_expr)

    def reaction_to_fluxes(self):
        """
        Convert reactions to fluxes -
        in general, for each product and each reactant there are two fluxes,
        one forward flux (dictated by :code:`self.eqn_f_str`)
        and one reverse flux (dictated by :code:`self.eqn_r_str`),
        stoichiometry is dictated by the number of times a given species occurs on the lhs or rhs
        """
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
    """Flux objects are created from reaction objects and should not be
    explicitly initialized by the user.
    Each flux object contains:

    * name: string (created as reaction name + species name + (f) or (r))
    * destination_species: flux increases or decreases this species
    * equation: directionality * stoichiometry * reaction string
    * reaction: reaction object this flux comes from
    """

    name: str
    destination_species: Species
    equation: sym.Expr
    reaction: Reaction

    def check_validity(self):
        "No validity checks for flux objects currently"
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
            modules=common.smart_expressions(gset["dolfin_expressions"]),
        )

        # Update equation with correct unit scale factor
        self._post_init_get_flux_units()
        self._post_init_get_integration_measure()

        # Check if flux is linear with respect to different components
        self.is_linear_wrt_comp = dict()
        self._post_init_get_is_linear_comp()

    def _post_init_get_involved_species_parameters_compartments(self):
        "Find species, parameters, and compartments involved in this flux"
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
        "Flux topology" refers to what types of compartments the flux occurs over
        For instance, if a reaction occurs only in a volume, its topology is type "volume"
        or if it involves species moving from one volume to another (across a surface),
        it is of type "volume-surface_to_volume". Depending on the flux topology,
        the flux will contribute to the system either as a term in the PDE or as
        a boundary condition.

        The dimensionality indicated in the left brackets below refers to how many
        compartments are involved in the given flux (e.g. 3d = 3 compartments)

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
        """
        Check that flux units match expected type of units.
        For boundary conditions, the flux units should be
        (concentration_units / compartment_units) * diffusion_units
        For fluxes that contribute to the PDE terms, units should be
        (concentration_units / time)
        If the units dimensionality does not match the expected form, an error is thrown;
        if the dimensionality matches, but the units scaling is not 1.0
        (e.g. if we have a flux specified as nM/s, but concentration_units are defined as uM),
        then self.unit_scale_factor is set to the appropriate factor
        (1000 in the example where we need to convert nM -> uM)
        NOTE: All unit checks are completed using pint, which may not be compatible with
        certain functions such as sign().
        """
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
        Flux topologies (cf. definitions in _post_init_get_flux_topology above):
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

    def equation_lambda_eval(self, input_type="quantity"):
        """Evaluates the equation lambda function using either the quantity
        (value * units), the value, or the units.
        The values and units are evaluated separately and then combined
        because some expressions don't work well
        with pint quantity types.
        """
        # This is an attempt to make the equation lambda work with pint quantities
        # note - throws an error when it doesn't return a float
        # (happens when it returns 0 from sign function, for instance)
        self._equation_quantity = self.equation_lambda(**self.equation_variables)
        if input_type == "quantity":
            return self._equation_quantity
        elif input_type == "value":
            return self._equation_quantity.magnitude
        elif input_type == "units":
            return unit_to_quantity(self._equation_quantity.units)

    # Seems like setting this as a @property doesn't cause fenics to recompile

    @property
    def form(self):
        """-1 factor because terms are defined as if they were on the
        lhs of the equation :math:`F(u;v)=0`"""
        return (
            d.Constant(-1)
            * self.equation_lambda_eval(input_type="value")
            * self.destination_species.v
            * self.measure
        )

    @property
    def scalar_form(self):
        """
        Defines scalar form for given flux.
        If the destination species is a vector function,
        the assembled form will be a vector of size NDOF.
        """
        return (
            d.Constant(-1)
            * self.equation_lambda_eval(input_type="value")
            * self.destination_species.vscalar
            * self.measure
        )

    @property
    def assembled_flux(self):
        """Attempt to convert flux units to molecules_per_second for printing."""
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
        If the flux is linear in terms of a compartment vector (e.g.
        :code:`dj/du['pm']`),
        then sets :code:`self.is_lienar_wrt_comp[comp_name]` to True
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
    """Initialize empty containers (pc, sc, cc, and rc)"""
    pc = ParameterContainer()
    sc = SpeciesContainer()
    cc = CompartmentContainer()
    rc = ReactionContainer()
    return pc, sc, cc, rc


def sbmodel_from_locals(local_values):
    """Assemble containers from local variables"""
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
