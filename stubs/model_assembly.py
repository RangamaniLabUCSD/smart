"""
Classes for parameters, species, compartments, reactions, fluxes, and forms
Model class contains functions to efficiently solve a system
"""
import pdb
import re
from collections import Counter
from collections import OrderedDict as odict
from collections import defaultdict as ddict
from typing import Type, Any

import dolfin as d
import ufl
import mpi4py.MPI as pyMPI
import pandas
import petsc4py.PETSc as PETSc
from termcolor import colored

Print = PETSc.Sys.Print

from copy import Error, copy, deepcopy
from pprint import pprint

import numpy as np
import pint
from scipy.integrate import cumtrapz, solve_ivp
import sympy as sym
from sympy import Heaviside, lambdify, Symbol, integrate
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.numbers import E
from sympy.utilities.iterables import flatten
from tabulate import tabulate

import stubs
import stubs.common as common
from stubs.common import _fancy_print as fancy_print
from stubs import unit

import dataclasses
from dataclasses import dataclass

comm = d.MPI.comm_world
rank = comm.rank
size = comm.size
root = 0

# ====================================================
# ====================================================
# Base Classes
# ====================================================
# ====================================================
class ObjectContainer:
    """
    Parent class containing general methods used by all "containers"
    """
    def __init__(self, ObjectClass):#df=None, Dict=None):
        self.Dict = odict()
        self._ObjectClass = ObjectClass
        self.properties_to_print = [] # properties to print

    # ==============================================================================
    # ObjectContainer - General methods
    # ------------------------------------------------------------------------------
    # General properties/methods to emulate some dictionary-like structure
    # + add some misc useful structure
    # ==============================================================================
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
                    iterator = iter(data)
                except:
                    raise TypeError("Data being added to ObjectContainer must be either the ObjectClass or an iterator.")
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
    def get_property(self, property_name):
        # returns a dict of properties
        property_dict = {}
        for key, obj in self.items:
            property_dict[key] = getattr(obj, property_name)
        return property_dict
    def get_index(self, idx):
        """
        Get an element of the object container ordered dict by referencing its index
        """
        return list(self.values)[idx]
    
    def sort_by(self, attribute: str, order='decreasing'):
        "Return a list of container's objects sorted by an attribute, and a list of the attribute values"
        attribute_values = [getattr(obj, attribute) for obj in self.values]
        ordering = np.argsort(attribute_values)
        if order=='decreasing':
            ordering = np.flip(ordering)

        return [self.get_index(idx) for idx in ordering], [attribute_values[idx] for idx in ordering]


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
            if properties_to_print and 'idx' not in properties_to_print:
                properties_to_print.insert(0, 'idx')
            for idx, (name, instance) in enumerate(self.items):
                df = df.append(instance.get_pandas_series(properties_to_print=properties_to_print, idx=idx))
        else:
            for idx, (name, instance) in enumerate(self.items):
                df = df.append(instance.get_pandas_series(properties_to_print=properties_to_print))
        # # sometimes types are recast. change entries into their original types
        # for dtypeName, dtype in self.dtypes.items():
        #     if dtypeName in df.columns:
        #         df = df.astype({dtypeName: dtype})

        return df

    def print_to_latex(self, properties_to_print=None, escape=False,
                       include_idx=False):
        df = self.get_pandas_dataframe(properties_to_print=properties_to_print,
                                       include_idx=include_idx)

        # converts Pint units to latex format
        unit_to_tex = lambda unit: '${:L}$'.format(unit._units)
        name_to_tex = lambda name: '$' + name.replace('_', '\_') + '$'
        # bit of a hack... check the first row to guess which columns contain
        # either Pint unit or quantity objects so we may convert them
        for name, value in list(df.iloc[0].iteritems()):
            if hasattr(value, '_units'):
                df[name] = df[name].apply(unit_to_tex)
            if name == 'name':
                df[name] = df[name].apply(name_to_tex)
            if name == 'idx' and include_idx == False:
                df = df.drop('idx',axis=1)

        with pandas.option_context("max_colwidth", 1000):
            tex_str = df.to_latex(index=False, longtable=True, escape=escape)
            print(tex_str)

    def print(self, tablefmt='fancy_grid', properties_to_print=None):
        if rank == root:
            if properties_to_print:
                if type(properties_to_print) != list: properties_to_print=[properties_to_print]
            elif hasattr(self, 'properties_to_print'):
                properties_to_print = self.properties_to_print
            df = self.get_pandas_dataframe(properties_to_print=properties_to_print)
            if properties_to_print:
                df = df[properties_to_print]

            print(tabulate(df, headers='keys', tablefmt=tablefmt))#,
                   #headers='keys', tablefmt=tablefmt), width=120)
        else:
            pass

    def __str__(self):
        df = self.get_pandas_dataframe(properties_to_print=self.properties_to_print)
        df = df[self.properties_to_print]

        return tabulate(df, headers='keys', tablefmt='fancy_grid')

    def vprint(self, keyList=None, properties_to_print=None, print_all=False):
        # in order of priority: kwarg, container object property, else print all keys
        if rank == root:
            if keyList:
                if type(keyList) != list: keyList=[keyList]
            elif hasattr(self, 'keyList'):
                keyList = self.keyList
            else:
                keyList = list(self.keys)

            if properties_to_print:
                if type(properties_to_print) != list: properties_to_print=[properties_to_print]
            elif hasattr(self, 'properties_to_print'):
                properties_to_print = self.properties_to_print

            if print_all: properties_to_print = []
            for key in keyList:
                self[key].print(properties_to_print=properties_to_print)
        else:
            pass


class ObjectInstance:
    """
    Parent class containing general methods used by all stubs
    "objects": i.e. parameters, species, compartments, reactions, fluxes, forms
    """

    def _convert_pint_quantity_to_unit(self):
        # strip the magnitude and keep the units. Warn if magnitude!=1
        for name, attr in vars(self).items():
            if isinstance(attr, pint.Quantity):
                setattr(self, name, common.pint_quantity_to_unit(attr))
        
    def _check_input_type_validity(self):
        "Check that the inputs have the same type (or are convertible) to the type hint."
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.type == Any:
                continue
            elif not isinstance(value, field.type):
                try:
                    setattr(self, field.name, field.type(value))
                except:
                    raise TypeError(f"Object \"{self.name}\" type error: the attribute \"{field.name}\" is expected to be {field.type}, got {type(value)} instead. "
                                    f"Conversion to the expected type was attempted but unsuccessful.")
    def _convert_pint_unit_to_quantity(self):
        # convert pint units to quantity
        for name, attr in vars(self).items():
            if isinstance(attr, pint.Unit):
                setattr(self, name, common.pint_unit_to_quantity(attr))

    def get_pandas_series(self, properties_to_print=None, idx=None):
        if properties_to_print:
            dict_to_convert = odict({'idx': idx})
            dict_to_convert.update(odict([(key,val) for (key,val) in self.__dict__.items() if key in properties_to_print]))
        else:
            dict_to_convert = self.__dict__
        return pandas.Series(dict_to_convert, name=self.name)
    def print(self, properties_to_print=None):
        if rank==root:
            print("Name: " + self.name)
            # if a custom list of properties to print is provided, only use those
            if properties_to_print:
                dict_to_print = dict([(key,val) for (key,val) in self.__dict__.items() if key in properties_to_print])
            else:
                dict_to_print = self.__dict__
            pprint(dict_to_print, width=240)
        else:
            pass


# ==============================================================================
# ==============================================================================
# Classes for parameters, species, compartments, reactions, and fluxes
# ==============================================================================
# ==============================================================================

class ParameterContainer(ObjectContainer):
    def __init__(self):
        super().__init__(Parameter)

        self.properties_to_print = ['value', 'unit', 'is_time_dependent', 'sym_expr', 'notes', 'group']

@dataclass
class Parameter(ObjectInstance):
    name: str
    value: float
    unit: pint.Unit
    group: str=''
    notes: str=''
    use_preintegration: bool=False

    @classmethod
    def from_file(cls, name, sampling_file, unit, group='', notes='', use_preintegration=False):
        "Load in a purely time-dependent scalar function from data"
        # load in sampling data file
        sampling_data = np.genfromtxt(sampling_file, dtype='float', delimiter=',')
        fancy_print(f"Loading in data for parameter {name}", format_type='log')

        # Only t0=0.0
        if sampling_data[0,0] != 0.0 or sampling_data.shape[1] != 2:
            raise NotImplementedError
        value = sampling_data[0,1] # initial value

        # Print("Creating dolfin object for time-dependent parameter %s" % self.name)
        parameter = cls(name, value, unit, group=group, notes=notes, use_preintegration=use_preintegration)

        if use_preintegration:
            # preintegrate sampling data
            int_data = cumtrapz(sampling_data[:,1], x=sampling_data[:,0], initial=0)
            # concatenate time vector
            preint_sampling_data = common.np_smart_hstack(sampling_data[:,0], int_data)
            parameter.preint_sampling_data  = preint_sampling_data

        # initialize instance
        parameter.sampling_data         = sampling_data
        parameter.is_time_dependent     = True
        parameter.is_space_dependent    = False # not supported yet
        parameter.type = 'from_file'
        fancy_print(f"Time-dependent parameter {name} loaded from file.", format_type='log')

        return parameter
    
    @classmethod
    def from_expression(cls, name, sym_expr, unit, preint_sym_expr=None, group='', notes='', use_preintegration=False):
        # Parse the given string to create a sympy expression
        sym_expr = parse_expr(sym_expr).subs({'x': 'x[0]', 'y': 'x[1]', 'z': 'x[2]'})
        
        # Check if expression is time/space dependent
        free_symbols = [str(x) for x in sym_expr.free_symbols]
        is_time_dependent  = 't' in free_symbols
        is_space_dependent = not {'x[0]','x[1]','x[2]'}.isdisjoint(set(free_symbols))
        if is_space_dependent:
            raise NotImplementedError
        # For now, parameters can only be defined in terms of time/space
        if not {'x[0]', 'x[1]', 'x[2]', 't'}.issuperset(free_symbols):
            raise NotImplementedError
        
        # TODO: fix this when implementing space dependent parameters
        if is_time_dependent:
            value = sym_expr.subs({'t': 0.0}).evalf()

        parameter = cls(name, value, unit, group=group, notes=notes, use_preintegration=use_preintegration)

        if use_preintegration:
            if preint_sym_expr:
                preint_sym_expr = parse_expr(preint_sym_expr).subs({'x': 'x[0]', 'y': 'x[1]', 'z': 'x[2]'})
            else:
                # try to integrate
                t = Symbol('t')
                preint_sym_expr = integrate(sym_expr, t)
            parameter.preint_sym_expr       = preint_sym_expr

        parameter.free_symbols          = free_symbols
        parameter.sym_expr              = sym_expr
        parameter.is_time_dependent     = is_time_dependent
        parameter.is_space_dependent    = is_space_dependent

        # parameter.dolfin_expression = d.Expression(sym.printing.ccode(sym_expr), t=0.0, degree=1)
        parameter.type == 'expression'
        fancy_print(f"Time-dependent parameter {name} evaluated from expression.", format_type='log')

        return parameter

    def __post_init__(self):
        if not hasattr(self, 'is_time_dependent'):
            self.is_time_dependent     = False
        if not hasattr(self, 'is_space_dependent'):
            self.is_space_dependent    = False

        if self.use_preintegration:
            fancy_print(f"Warning! Pre-integrating parameter {self.name}. Make sure that expressions {self.name} appears in have no other time-dependent variables.", format_type='warning')
        
        attributes = ['sym_expr', 'preint_sym_expr', 'sampling_data', 'preint_sampling_data']
        for attribute in attributes:
            if not hasattr(self, attribute):
                setattr(self, attribute, None)
        
        if not hasattr(self, 'type'):
            self.type = 'constant'

        self._convert_pint_quantity_to_unit()
        self._check_input_type_validity()
        self._convert_pint_unit_to_quantity()
        self.check_validity()

        #self.dolfin_constant = d.Constant(self.value)
        #self.value_unit = self.value*self.unit


    @property
    def dolfin_quantity(self):
        if hasattr(self, 'dolfin_expression'):
            return self.dolfin_expression * self.unit
        else:
            return self.dolfin_constant * self.unit
    
    @property
    def quantity(self):
        return self.value * self.unit

    def check_validity(self):
        if self.is_time_dependent:
            if all([x=='' for x in [self.sampling_file, self.sym_expr, self.preint_sym_expr]]):
                raise ValueError(f"Parameter {self.name} is marked as time dependent but is not defined in terms of time.")


class SpeciesContainer(ObjectContainer):
    def __init__(self):
        super().__init__(Species)

        self.properties_to_print = ['compartment_name', 'dof_index', 'concentration_units', 'D', 'initial_condition', 'group']

    # def assemble_dolfin_functions(self, rc, cc):
    #     """
    #     define dof/solution vectors (dolfin trialfunction, testfunction, and function types) based on number of species appearing in reactions
    #     IMPORTANT: this function will create additional species on boundaries in order to use operator-splitting later on
    #     e.g.
    #     A [cyto] + B [pm] <-> C [pm]
    #     Since the value of A is needed on the pm there will be a species A_b_pm which is just the values of A on the boundary pm
    #     """

    #     # functions to run beforehand as we need their results
    #     num_species_per_compartment = rc.get_species_compartment_counts(self, cc)
    #     #cc.get_min_max_dim() # refactor
    #     self.assemble_compartment_indices(rc, cc)
    #     cc.add_property_to_all('is_in_a_reaction', False)
    #     cc.add_property_to_all('V', None)

    #     V, u, v = {}, {}, {}
    #     for compartment_name, num_species in num_species_per_compartment.items():
    #         compartmentDim = cc[compartment_name].dimensionality
    #         cc[compartment_name].num_species = num_species
    #         if rank==root:
    #             print('Compartment %s (dimension: %d) has %d species associated with it' %
    #                   (compartment_name, compartmentDim, num_species))

    #         # u is the actual function. t is for linearized versions. k is for picard iterations. n is for last time-step solution
    #         if num_species == 1:
    #             V[compartment_name] = d.FunctionSpace(cc.meshes[compartment_name], 'P', 1)
    #             u[compartment_name] = {'u': d.Function(V[compartment_name], name="concentration_u"), 't': d.TrialFunction(V[compartment_name]),
    #             'k': d.Function(V[compartment_name]), 'n': d.Function(V[compartment_name])}
    #             v[compartment_name] = d.TestFunction(V[compartment_name])
    #         else: # vector space
    #             V[compartment_name] = d.VectorFunctionSpace(cc.meshes[compartment_name], 'P', 1, dim=num_species)
    #             u[compartment_name] = {'u': d.Function(V[compartment_name], name="concentration_u"), 't': d.TrialFunctions(V[compartment_name]),
    #             'k': d.Function(V[compartment_name]), 'n': d.Function(V[compartment_name])}
    #             v[compartment_name] = d.TestFunctions(V[compartment_name])

    #     # now we create boundary functions, i.e. interpolations of functions defined on the volume
    #     # to function spaces of the surrounding mesh
    #     V['boundary'] = {}
    #     for compartment_name, num_species in num_species_per_compartment.items():
    #         compartmentDim = cc[compartment_name].dimensionality
    #         if compartmentDim == cc.max_dim: # mesh may have boundaries
    #             V['boundary'][compartment_name] = {}
    #             for mesh_name, mesh in cc.meshes.items():
    #                 if compartment_name != mesh_name and mesh.topology().dim() < compartmentDim:
    #                     if num_species == 1:
    #                         boundaryV = d.FunctionSpace(mesh, 'P', 1)
    #                     else:
    #                         boundaryV = d.VectorFunctionSpace(mesh, 'P', 1, dim=num_species)
    #                     V['boundary'][compartment_name].update({mesh_name: boundaryV})
    #                     u[compartment_name]['b_'+mesh_name] = d.Function(boundaryV, name="concentration_ub")

    #     # now we create volume functions, i.e. interpolations of functions defined on the surface
    #     # to function spaces of the associated volume
    #     V['volume'] = {}
    #     for compartment_name, num_species in num_species_per_compartment.items():
    #         compartmentDim = cc[compartment_name].dimensionality
    #         if compartmentDim == cc.min_dim: # mesh may be a boundary with a connected volume
    #             V['volume'][compartment_name] = {}
    #             for mesh_name, mesh in cc.meshes.items():
    #                 if compartment_name != mesh_name and mesh.topology().dim() > compartmentDim:
    #                     if num_species == 1:
    #                         volumeV = d.FunctionSpace(mesh, 'P', 1)
    #                     else:
    #                         volumeV = d.VectorFunctionSpace(mesh, 'P', 1, dim=num_species)
    #                     V['volume'][compartment_name].update({mesh_name: volumeV})
    #                     u[compartment_name]['v_'+mesh_name] = d.Function(volumeV, name="concentration_uv")

    #     # associate indexed functions with dataframe
    #     for key, sp in self.items:
    #         sp.u = {}
    #         sp.v = None
    #         if sp.is_in_a_reaction:
    #             sp.compartment.is_in_a_reaction = True
    #             num_species = sp.compartment.num_species
    #             for key in u[sp.compartment_name].keys():
    #                 if num_species == 1:
    #                     sp.u.update({key: u[sp.compartment_name][key]})
    #                     sp.v = v[sp.compartment_name]
    #                 else:
    #                     sp.u.update({key: u[sp.compartment_name][key][sp.dof_index]})
    #                     sp.v = v[sp.compartment_name][sp.dof_index]

    #     # # associate function spaces with dataframe
    #     for key, comp in cc.items:
    #         if comp.is_in_a_reaction:
    #             comp.V = V[comp.name]

    #     self.u = u
    #     self.v = v
    #     self.V = V

@dataclass
class Species(ObjectInstance):
    name: str
    initial_condition: Any
    #initial_condition: float
    concentration_units: pint.Unit
    D: float
    diffusion_units: pint.Unit
    compartment_name: str
    group: str=''

    def __post_init__(self):
        self.sub_species = {} # additional compartments this species may live in in addition to its primary one
        self.is_in_a_reaction = False
        self.is_an_added_species = False
        self.dof_map = None
        self.u       = dict()
        self._usplit = dict()
        self.ut      = None
        self.v       = None
        #self.t       = 0.0

        if isinstance(self.initial_condition, int):
            self.initial_condition = float(self.initial_condition)
        elif isinstance(self.initial_condition, str):
            # Parse the given string to create a sympy expression
            sym_expr = parse_expr(self.initial_condition).subs({'x': 'x[0]', 'y': 'x[1]', 'z': 'x[2]'})
            
            # Check if expression is space dependent
            free_symbols = [str(x) for x in sym_expr.free_symbols]
            if not {'x[0]', 'x[1]', 'x[2]'}.issuperset(free_symbols):
                raise NotImplementedError
            fancy_print(f"Creating dolfin object for space-dependent initial condition {self.name}", format_type='log')
            self.initial_condition_expression = d.Expression(sym.printing.ccode(sym_expr), degree=1)
        else:
            raise TypeError(f"initial_condition must be a float or string.")
        
        self._convert_pint_quantity_to_unit()
        self._check_input_type_validity()
        self._convert_pint_unit_to_quantity()
        self.check_validity()

    def check_validity(self):
        # checking values
        if isinstance(self.initial_condition, float) and self.initial_condition < 0.0:
            raise ValueError(f"Initial condition for species {self.name} must be greater or equal to 0.")
        if self.D < 0.0:
            raise ValueError(f"Diffusion coefficient for species {self.name} must be greater or equal to 0.")
        # checking units
        if not self.diffusion_units.check('[length]^2/[time]'):
            raise ValueError(f"Units of diffusion coefficient for species {self.name} must be dimensionally equivalent to [length]^2/[time].")
        if not any([self.concentration_units.check(f'mole/[length]^{dim}') for dim in [1,2,3]]):
            raise ValueError(f"Units of concentration for species {self.name} must be dimensionally equivalent to mole/[length]^dim where dim is either 1, 2, or 3.")

    @property
    def dolfin_quantity(self):
        return self._usplit['u'] * self.concentration_units

    @property
    def D_quantity(self):
        return self.D * self.diffusion_units


class CompartmentContainer(ObjectContainer):
    def __init__(self):
        super().__init__(Compartment)

        self.properties_to_print = ['_mesh_id', 'dimensionality', 'num_species', '_num_vertices', '_num_dofs', '_num_cells', 'cell_marker', '_nvolume']
    
    def print(self, tablefmt='fancy_grid'):
        for c in self:
            c.mesh_id
            c.nvolume
            c.num_vertices
            c.num_dofs
            c.num_cells
        super().print(tablefmt, self.properties_to_print)


@dataclass
class Compartment(ObjectInstance):
    name: str
    dimensionality: int
    compartment_units: pint.Unit
    cell_marker: Any
    
    def __post_init__(self):

        if isinstance(self.cell_marker, list) and not all([isinstance(m, int) for m in self.cell_marker]) \
            or not isinstance(self.cell_marker, (int,list)):
                raise TypeError(f"cell_marker must be an int or list of ints.")

        self._convert_pint_quantity_to_unit()
        self._check_input_type_validity()
        self._convert_pint_unit_to_quantity()
        self.check_validity()

        # Initialize
        self.species = odict()
        self.u       = dict()
        self._usplit = dict()
        self.V       = None
        self.v       = None
    
    def check_validity(self):
        if self.dimensionality not in [1,2,3]:
            raise ValueError(f"Compartment {self.name} has dimensionality {self.dimensionality}. Dimensionality must be in [1,2,3].")
        # checking units
        if not self.compartment_units.check('[length]'):
            raise ValueError(f"Compartment {self.name} has units of {self.compartment_units} - units must be dimensionally equivalent to [length].")
    
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
        self._nvolume = self.mesh.nvolume * self.compartment_units ** self.dimensionality
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
        self._num_dofs = self.num_species * self.num_vertices
        return self._num_dofs


class ReactionContainer(ObjectContainer):
    def __init__(self):
        super().__init__(Reaction)

        #self.properties_to_print = ['name', 'lhs', 'rhs', 'eqn_f', 'eqn_r', 'param_map', 'reaction_type', 'explicit_restriction_to_domain', 'group']
        self.properties_to_print = ['lhs', 'rhs', 'eqn_f_str', 'eqn_r_str']

@dataclass
class Reaction(ObjectInstance):
    name: str
    lhs: list
    rhs: list
    param_map: dict
    species_map: dict = dataclasses.field(default_factory=dict)
    reaction_type: str='mass_action'
    explicit_restriction_to_domain: str=''
    track_value: bool=False
    eqn_f_str: str=''
    eqn_r_str: str=''

    def __post_init__(self):
        self._check_input_type_validity()
        self.check_validity()
        self.fluxes = dict()

        # Finish initializing the species map
        for species_name in set(self.lhs + self.rhs):
            if species_name not in self.species_map:
                self.species_map[species_name] = species_name

    def check_validity(self):
        # Type checking
        if not all([isinstance(x, str) for x in self.lhs]):
            raise TypeError(f"Reaction {self.name} requires a list of strings as input for lhs.")
        if not all([isinstance(x, str) for x in self.rhs]):
            raise TypeError(f"Reaction {self.name} requires a list of strings as input for rhs.")
        if not all([type(k)==str and type(v)==str for (k,v) in self.param_map.items()]):
            raise TypeError(f"Reaction {self.name} requires a dict of str:str as input for param_map.")
        if self.species_map:
            if not all([k==str and v==str for (k,v) in self.species_map.items()]):
                raise TypeError(f"Reaction {self.name} requires a dict of str:str as input for species_map.")
            
    def _parse_custom_reaction(self, reaction_eqn_str):
        reaction_expr = parse_expr(reaction_eqn_str)
        reaction_expr = reaction_expr.subs(self.param_map)
        reaction_expr = reaction_expr.subs(self.species_map)
        return str(reaction_expr)

    def reaction_to_fluxes(self):
        # set of 2-tuples. (species_name, signed stoichiometry)
        species_stoich      = {(species_name, -1*self.lhs.count(species_name)) for species_name in self.lhs}
        species_stoich.update({(species_name,  1*self.rhs.count(species_name)) for species_name in self.rhs})

        for species_name, stoich in species_stoich:
            species = self.species[species_name]
            if self.eqn_f_str:
                flux_name     = self.name + f" [{species_name} (f)]"
                eqn           = stoich * parse_expr(self.eqn_f_str)
                self.fluxes.update({flux_name: Flux(flux_name, species, eqn, self)})
            if self.eqn_r_str:
                flux_name     = self.name + f" [{species_name} (r)]"
                eqn           = -stoich * parse_expr(self.eqn_r_str)
                self.fluxes.update({flux_name: Flux(flux_name, species, eqn, self)})


class FluxContainer(ObjectContainer):
    def __init__(self):
        super().__init__(Flux)

        # self.properties_to_print = ['species_name', 'equation', 'sign', 'involved_species',
        #                      'involved_parameters', 'source_compartment',
        #                      'destination_compartment', 'ukeys', 'group']

        self.properties_to_print = ['_species_name', 'equation', 'topology', 'equation_quantity']#, 'ukeys']#'source_compartment', 'destination_compartment', 'ukeys']

@dataclass
class Flux(ObjectInstance):
    name: str
    destination_species: Species
    equation: sym.Expr
    reaction: Reaction
    #signed_stoich: int
    # species_map: dict
    # param_map: dict
    # #group: str
    # parent_reaction: Reaction
    # explicit_restriction_to_domain: str=''
    # track_value: bool=False        
    
    def check_validity(self):
        pass

    def __post_init__(self):
        # self.tracked_values = []
        # for nice printing
        self._species_name = self.destination_species.name

        self.surface_id = set()
        self.volume_ids = set()

        self._check_input_type_validity()
        self.check_validity()

        # Add in an uninitialized unit_scale_factor
        self.unit_scale_factor = 1.0*unit.dimensionless
        self.equation = self.equation * Symbol('unit_scale_factor')

        # Getting additional flux properties
        self._post_init_get_involved_species_parameters_compartments()
        self._post_init_get_flux_topology()
        # Evaluate equation with no unit scale factor
        self._post_init_get_lambda_equation()
        self.evaluate_equation()
        # Update equation with correct unit scale factor
        self._post_init_get_flux_units()
        self._post_init_get_integration_measure()

    def _post_init_get_involved_species_parameters_compartments(self):
        self.destination_compartment = self.destination_species.compartment
        
        # Get the subset of species/parameters/compartments that are relevant
        variables = {str(x) for x in self.equation.free_symbols}
        all_params  = self.reaction.parameters
        all_species = self.reaction.species
        self.parameters    = {x: all_params[x] for x in variables.intersection(all_params.keys())}
        self.species       = {x: all_species[x] for x in variables.intersection(all_species.keys())}
        self.compartments  = self.reaction.compartments

    
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
        if self.reaction.topology in ['surface', 'volume']:
            self.topology = self.reaction.topology
            source_compartments = {self.destination_compartment.name}
        # 2 or 3 compartment flux
        elif self.reaction.topology in ['volume_surface', 'volume_surface_volume']:
            source_compartments = set(self.compartments.keys()).difference({self.destination_compartment.name})
            
            if self.reaction.topology == 'volume_surface':
                assert len(source_compartments) == 1
                if self.destination_compartment.is_volume:
                    self.topology = 'surface_to_volume'
                else:
                    self.topology = 'volume_to_surface'

            elif self.reaction.topology == 'volume_surface_volume':
                assert len(source_compartments) == 2
                if self.destination_compartment.is_volume:
                    self.topology = 'volume-surface_to_volume'
                else:
                    self.topology = 'volume-volume_to_surface'

        else:
            raise AssertionError()
            
        self.source_compartments = {name: self.reaction.compartments[name] for name in source_compartments}
        self.surface = [c for c in self.compartments.values() if c.mesh.is_surface]
        if len(self.surface) == 1:
            self.surface = self.surface[0]
        else:
            self.surface = None
        self.volumes = [c for c in self.compartments.values() if c.mesh.is_volume]
        self.surface_id = frozenset([c.mesh.id for c in self.compartments.values() if c.mesh.is_surface])
        self.volume_ids = frozenset([c.mesh.id for c in self.compartments.values() if c.mesh.is_volume])

        # Based on topology we know if it is a boundary condition or RHS term
        if self.topology in ['volume', 'surface', 'volume_to_surface', 'volume-volume_to_surface']:
            self.is_boundary_condition = False
        elif self.topology in ['surface_to_volume', 'volume-surface_to_volume']:
            self.is_boundary_condition = True
        else:
            raise AssertionError()
    
    def _post_init_get_lambda_equation(self):
        self.equation_lambda = sym.lambdify(list(self.equation_variables.keys()), self.equation, modules=['sympy','numpy'])

    def _post_init_get_flux_units(self):
        concentration_units = self.destination_species.concentration_units
        compartment_units   = self.destination_compartment.compartment_units
        diffusion_units     = self.destination_species.diffusion_units

        # The expected units
        if self.is_boundary_condition:
            self._flux_units = concentration_units / compartment_units * diffusion_units # ~D*du/dn
        else:
            self._flux_units = concentration_units / unit.s # rhs term. ~du/dt

        # If unit dimensionality is not correct a parameter likely needs to be adjusted
        if self._flux_units.dimensionality != self.equation_units.dimensionality:
            raise ValueError(f"Flux {self.name} has wrong units "
                                f"(expected {self._flux_units}, got {self.equation_units}.")
        # Fix scaling 
        else:
            self.unit_scale_factor = self.equation_units.to(self._flux_units)/self.equation_units
            assert self.unit_scale_factor.dimensionless
            assert self.equation_units*self.unit_scale_factor == self.equation_units.to(self._flux_units)

            if self.unit_scale_factor.magnitude == 1.0:
                return

            fancy_print(f"\nFlux {self.name} scaled by {self.unit_scale_factor}", format_type='log')
            fancy_print(f"Old flux units: {self.equation_units}", format_type='log')
            fancy_print(f"New flux units: {self._flux_units}", format_type='log')
            print("")

            # update equation with new scale factor
            self._post_init_get_lambda_equation()
            self.evaluate_equation()
            assert self.equation_units == self._flux_units

    def _post_init_get_integration_measure(self):
        """
        Flux topologys:
        [1d] volume:                    PDE of u
        [1d] surface:                   PDE of v
        [2d] volume_to_surface:         PDE of v            
        [2d] surface_to_volume:         BC of u
        [3d] volume-surface_to_volume:  BC of u ()
        [3d] volume-volume_to_surface:  PDE of v ()
    
        1d means no other compartment are involved, so the integration measure is the volume of the compartment
        2d/3d means two/three compartments are involved, so the integration measure is the intersection between all compartments
        """
        #if not self.is_boundary_condition:
        if self.topology in ['volume', 'surface']:
            self.measure       = self.destination_compartment.mesh.dx
        elif self.topology in ['volume_to_surface', 'surface_to_volume', 'volume-volume_to_surface', 'volume-surface_to_volume']:
            # intersection of this surface with boundary of source volume(s)
            assert self.surface.mesh.has_intersection[self.volume_ids] # make sure there is at least one entity with all compartments involved
            #assert self.has_intersection[self.volume_ids] 
            self.measure = self.surface.mesh.dx_map[self.volume_ids](1)

    @property
    def equation_variables(self):
        variables = {variable.name: variable.dolfin_quantity for variable in {**self.parameters, **self.species}.values()}
        variables.update({'unit_scale_factor': self.unit_scale_factor})
        return variables
    
    @property
    def equation_value(self):
        return self.equation_quantity.magnitude
    @property
    def equation_units(self):
        return common.pint_unit_to_quantity(self.equation_quantity.units)
        
    def evaluate_equation(self):
        "Updates equation_value and equation_units"
        self.equation_quantity  = self.equation_lambda(**self.equation_variables)
    
    @property
    def form(self):
        "-1 factor because terms are defined as if they were on the lhs of the equation F(u;v)=0"
        self.evaluate_equation()
        form_result = d.Constant(-1) * self.equation_value * self.destination_species.v * self.measure
        self._form = form_result
        return form_result

    # def get_is_linear(self):
    #     """
    #     For a given flux we want to know which terms are linear
    #     """
    #     is_linear_wrt = {}
    #     for sym_var in self.sym_list:
    #         var_name = str(sym_var)
    #         if var_name in self.involved_species:
    #             if sym.diff(self.equation, var_name , 2).is_zero:
    #                 is_linear_wrt[var_name] = True
    #             else:
    #                 is_linear_wrt[var_name] = False

    #     self.is_linear_wrt = is_linear_wrt

    # def get_is_linear_comp(self):
    #     """
    #     Is the flux linear in terms of a compartment vector (e.g. dj/du['pm'])
    #     """
    #     is_linear_wrt_comp = {}
    #     umap = {}

    #     for var_name in self.sym_list:
    #         if var_name in self.involved_species:
    #             comp_name = self.species_map[var_name].compartment_name
    #             umap.update({var_name: 'u'+comp_name})

    #     new_eqn = self.equation.subs(umap)

    #     for comp_name in self.involved_compartments:
    #         if sym.diff(new_eqn, 'u'+comp_name, 2).is_zero:
    #             is_linear_wrt_comp[comp_name] = True
    #         else:
    #             is_linear_wrt_comp[comp_name] = False

    #     self.is_linear_wrt_comp = is_linear_wrt_comp


class FormContainer(ObjectContainer):
    def __init__(self):
        super().__init__(Form)

        self.properties_to_print = ['form', 'form_type', '_compartment_name']
    
    def print(self, tablefmt='fancy_grid', properties_to_print=None):
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
    """
    name: str
    form: ufl.Form
    species: Species
    form_type: str
    is_lhs: bool

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
        pass

    @property
    def integrals(self):
        self._integrals = self.form.integrals()
        return self._integrals

    def inspect(self):
        for index, integral in enumerate(self.integrals):
            print(str(integral) + "\n")
