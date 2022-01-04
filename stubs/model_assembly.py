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
import mpi4py.MPI as pyMPI
import pandas
import petsc4py.PETSc as PETSc
from termcolor import colored
from ufl.operators import variable

Print = PETSc.Sys.Print

from copy import copy, deepcopy
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

color_print = common.color_print

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
    def get_pandas_dataframe(self, properties_to_print=[], include_idx=True):
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

    def print_to_latex(self, properties_to_print=[], escape=False,
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

    def print(self, tablefmt='fancy_grid', properties_to_print=[]):
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

    def vprint(self, keyList=None, properties_to_print=[], print_all=False):
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

    def get_pandas_series(self, properties_to_print=[], idx=None):
        if properties_to_print:
            dict_to_convert = odict({'idx': idx})
            dict_to_convert.update(odict([(key,val) for (key,val) in self.__dict__.items() if key in properties_to_print]))
        else:
            dict_to_convert = self.__dict__
        return pandas.Series(dict_to_convert, name=self.name)
    def print(self, properties_to_print=[]):
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

        self.properties_to_print = ['name', 'value', 'unit', 'is_time_dependent', 'sym_expr', 'notes', 'group']

@dataclass
class Parameter(ObjectInstance):
    name: str
    value: float
    unit: pint.Unit
    group: str=''
    notes: str=''

    @classmethod
    def from_file(cls, name, sampling_file, unit, group='', notes=''):
        "Load in a purely time-dependent scalar function from data"
        # load in sampling data file
        sampling_data = np.genfromtxt(sampling_file, dtype='float', delimiter=',')
        fancy_print(f"Loading in data for parameter {name}", format_type='log')
        if sampling_data[0,0] != 0.0 or sampling_data.shape[1] != 2:
            raise NotImplementedError
        value = sampling_data[0,1] # initial value

        # Print("Creating dolfin object for time-dependent parameter %s" % self.name)

        # preintegrate sampling data
        int_data = cumtrapz(sampling_data[:,1], x=sampling_data[:,0], initial=0)
        # concatenate time vector
        preint_sampling_data = common.np_smart_hstack(sampling_data[:,0], int_data)

        # initialize instance
        parameter = cls(name, value, unit, group=group, notes=notes)
        parameter.sampling_data         = sampling_data
        parameter.preint_sampling_data  = preint_sampling_data
        parameter.is_time_dependent     = True
        parameter.is_space_dependent    = False # not supported yet
        fancy_print(f"Creating dolfin object for time-dependent parameter {name}", format_type='log')
        parameter.dolfin_constant = d.Constant(value)

        return parameter
    
    @classmethod
    def from_expression(cls, name, sym_expr, unit, preint_sym_expr=None, group='', notes=''):
        # Parse the given string to create a sympy expression
        sym_expr = parse_expr(sym_expr).subs({'x': 'x[0]', 'y': 'x[1]', 'z': 'x[2]'})
        
        # Check if expression is time/space dependent
        free_symbols = [str(x) for x in sym_expr.free_symbols]
        is_time_dependent  = 't' in free_symbols
        is_space_dependent = not {'x[0]','x[1]','x[2]'}.isdisjoint(set(free_symbols))
        if is_space_dependent:
            raise NotImplementedError
        if not {'x[0]', 'x[1]', 'x[2]', 't'}.issuperset(free_symbols):
            raise NotImplementedError
        
        # fix this when implementing space dependent parameters
        if is_time_dependent:
            value = sym_expr.subs({'t': 0.0}).evalf()

        if preint_sym_expr:
            preint_sym_expr = parse_expr(preint_sym_expr).subs({'x': 'x[0]', 'y': 'x[1]', 'z': 'x[2]'})
        elif is_time_dependent:
            # try to integrate
            t = Symbol('t')
            preint_sym_expr = integrate(sym_expr, t)

        parameter = cls(name, value, unit, group=group, notes=notes)
        parameter.free_symbols          = free_symbols
        parameter.sym_expr              = sym_expr
        parameter.preint_sym_expr       = preint_sym_expr
        parameter.is_time_dependent     = is_time_dependent
        parameter.is_space_dependent    = is_space_dependent

        fancy_print(f"Creating dolfin object for time-dependent parameter {name}", format_type='log')
        parameter.dolfin_constant = d.Constant(value)
        parameter.dolfin_expression = d.Expression(sym.printing.ccode(sym_expr), t=0.0, degree=1)

        return parameter

    def __post_init__(self):
        if not hasattr(self, 'is_time_dependent'):
            self.is_time_dependent     = False
        if not hasattr(self, 'is_space_dependent'):
            self.is_space_dependent    = False

        self.t = 0.0

        self._convert_pint_quantity_to_unit()
        self._check_input_type_validity()
        self._convert_pint_unit_to_quantity()
        self.check_validity()

        self.value_unit = self.value*self.unit

    def check_validity(self):
        if self.is_time_dependent:
            if all([x=='' for x in [self.sampling_file, self.sym_expr, self.preint_sym_expr]]):
                raise ValueError(f"Parameter {self.name} is marked as time dependent but is not defined in terms of time.")


class SpeciesContainer(ObjectContainer):
    def __init__(self):
        super().__init__(Species)

        self.properties_to_print = ['name', 'compartment_name', 'dof_index', 'concentration_units', 'D', 'initial_condition', 'group']

    # def assemble_compartment_indices(self, rc, cc):
    #     """
    #     Adds a column to the species dataframe which indicates the index of a species relative to its compartment
    #     """
    #     num_species_per_compartment = rc.get_species_compartment_counts(self, cc)
    #     for compartment, num_species in num_species_per_compartment.items():
    #         idx = 0
    #         comp_species = [sp for sp in self.values if sp.compartment_name==compartment]
    #         for sp in comp_species:
    #             if sp.is_in_a_reaction:
    #                 sp.dof_index = idx
    #                 idx += 1
    #             else:
    #                 print('Warning: species %s is not used in any reactions!' % sp.name)

    def assemble_dolfin_functions(self, rc, cc):
        """
        define dof/solution vectors (dolfin trialfunction, testfunction, and function types) based on number of species appearing in reactions
        IMPORTANT: this function will create additional species on boundaries in order to use operator-splitting later on
        e.g.
        A [cyto] + B [pm] <-> C [pm]
        Since the value of A is needed on the pm there will be a species A_b_pm which is just the values of A on the boundary pm
        """

        # functions to run beforehand as we need their results
        num_species_per_compartment = rc.get_species_compartment_counts(self, cc)
        #cc.get_min_max_dim() # refactor
        self.assemble_compartment_indices(rc, cc)
        cc.add_property_to_all('is_in_a_reaction', False)
        cc.add_property_to_all('V', None)

        V, u, v = {}, {}, {}
        for compartment_name, num_species in num_species_per_compartment.items():
            compartmentDim = cc[compartment_name].dimensionality
            cc[compartment_name].num_species = num_species
            if rank==root:
                print('Compartment %s (dimension: %d) has %d species associated with it' %
                      (compartment_name, compartmentDim, num_species))

            # u is the actual function. t is for linearized versions. k is for picard iterations. n is for last time-step solution
            if num_species == 1:
                V[compartment_name] = d.FunctionSpace(cc.meshes[compartment_name], 'P', 1)
                u[compartment_name] = {'u': d.Function(V[compartment_name], name="concentration_u"), 't': d.TrialFunction(V[compartment_name]),
                'k': d.Function(V[compartment_name]), 'n': d.Function(V[compartment_name])}
                v[compartment_name] = d.TestFunction(V[compartment_name])
            else: # vector space
                V[compartment_name] = d.VectorFunctionSpace(cc.meshes[compartment_name], 'P', 1, dim=num_species)
                u[compartment_name] = {'u': d.Function(V[compartment_name], name="concentration_u"), 't': d.TrialFunctions(V[compartment_name]),
                'k': d.Function(V[compartment_name]), 'n': d.Function(V[compartment_name])}
                v[compartment_name] = d.TestFunctions(V[compartment_name])

        # now we create boundary functions, i.e. interpolations of functions defined on the volume
        # to function spaces of the surrounding mesh
        V['boundary'] = {}
        for compartment_name, num_species in num_species_per_compartment.items():
            compartmentDim = cc[compartment_name].dimensionality
            if compartmentDim == cc.max_dim: # mesh may have boundaries
                V['boundary'][compartment_name] = {}
                for mesh_name, mesh in cc.meshes.items():
                    if compartment_name != mesh_name and mesh.topology().dim() < compartmentDim:
                        if num_species == 1:
                            boundaryV = d.FunctionSpace(mesh, 'P', 1)
                        else:
                            boundaryV = d.VectorFunctionSpace(mesh, 'P', 1, dim=num_species)
                        V['boundary'][compartment_name].update({mesh_name: boundaryV})
                        u[compartment_name]['b_'+mesh_name] = d.Function(boundaryV, name="concentration_ub")

        # now we create volume functions, i.e. interpolations of functions defined on the surface
        # to function spaces of the associated volume
        V['volume'] = {}
        for compartment_name, num_species in num_species_per_compartment.items():
            compartmentDim = cc[compartment_name].dimensionality
            if compartmentDim == cc.min_dim: # mesh may be a boundary with a connected volume
                V['volume'][compartment_name] = {}
                for mesh_name, mesh in cc.meshes.items():
                    if compartment_name != mesh_name and mesh.topology().dim() > compartmentDim:
                        if num_species == 1:
                            volumeV = d.FunctionSpace(mesh, 'P', 1)
                        else:
                            volumeV = d.VectorFunctionSpace(mesh, 'P', 1, dim=num_species)
                        V['volume'][compartment_name].update({mesh_name: volumeV})
                        u[compartment_name]['v_'+mesh_name] = d.Function(volumeV, name="concentration_uv")

        # associate indexed functions with dataframe
        for key, sp in self.items:
            sp.u = {}
            sp.v = None
            if sp.is_in_a_reaction:
                sp.compartment.is_in_a_reaction = True
                num_species = sp.compartment.num_species
                for key in u[sp.compartment_name].keys():
                    if num_species == 1:
                        sp.u.update({key: u[sp.compartment_name][key]})
                        sp.v = v[sp.compartment_name]
                    else:
                        sp.u.update({key: u[sp.compartment_name][key][sp.dof_index]})
                        sp.v = v[sp.compartment_name][sp.dof_index]

        # # associate function spaces with dataframe
        for key, comp in cc.items:
            if comp.is_in_a_reaction:
                comp.V = V[comp.name]

        self.u = u
        self.v = v
        self.V = V

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
        self.ut      = None
        self.v       = None
        self.v       = None
        self.t       = 0.0

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
            self.initial_condition_expression = d.Expression(sym.printing.ccode(sym_expr), t=0.0, degree=1)
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


class CompartmentContainer(ObjectContainer):
    def __init__(self):
        super().__init__(Compartment)

        self.properties_to_print = ['name', 'dimensionality', 'num_species', 'num_vertices', 'cell_marker', 'is_in_a_reaction', 'nvolume']
        self.meshes = {}
        self.vertex_mappings = {} # from submesh -> parent indices


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
        self.V       = dict()
        self.v       = dict()
    
    def check_validity(self):
        if self.dimensionality not in [1,2,3]:
            raise ValueError(f"Compartment {self.name} has dimensionality {self.dimensionality}. Dimensionality must be in [1,2,3].")
        # checking units
        if not self.compartment_units.check('[length]'):
            raise ValueError(f"Compartment {self.name} has units of {self.compartment_units} - units must be dimensionally equivalent to [length].")

    @property
    def dolfin_mesh(self):
        return self.mesh.dolfin_mesh

    @property
    def nvolume(self):
        "nvolume with proper units"
        return self.mesh.nvolume * self.compartment_units ** self.dimensionality
    
    @property
    def num_cells(self):
        return self.mesh.num_cells
    @property
    def num_facets(self):
        return self.mesh.num_facets
    @property
    def num_vertices(self):
        return self.mesh.num_vertices

    @property
    def num_dofs(self):
        "Number of degrees of freedom for this compartment"
        return self.num_species * self.num_vertices


class ReactionContainer(ObjectContainer):
    def __init__(self):
        super().__init__(Reaction)

        #self.properties_to_print = ['name', 'lhs', 'rhs', 'eqn_f', 'eqn_r', 'param_map', 'reaction_type', 'explicit_restriction_to_domain', 'group']
        self.properties_to_print = ['name', 'lhs', 'rhs', 'eqn_f']#, 'eqn_r']

    # def get_species_compartment_counts(self, sc, cc):
    #     """
    #     Returns a Counter object with the number of times a species appears in each compartment
    #     """
    #     self.do_to_all('get_involved_species_and_compartments', {"sc": sc, "cc": cc})
    #     all_involved_species = set([sp for species_set in [rxn.involved_species_link.values() for rxn in self.values] for sp in species_set])
    #     for sp_name, sp in sc.items:
    #         if sp in all_involved_species:
    #             sp.is_in_a_reaction = True

    #     compartment_counts = [sp.compartment_name for sp in all_involved_species]

    #     return Counter(compartment_counts)


    # def reaction_to_fluxes(self):
    #     self.do_to_all('reaction_to_fluxes')
    #     flux_list = []
    #     for rxn in self.values:
    #         for f in rxn.flux_list:
    #             flux_list.append(f)
    #     self.flux_list = flux_list
    # def get_flux_container(self):
    #     return FluxContainer(Dict=odict([(f.flux_name, f) for f in self.flux_list]))

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
                flux_name     = self.name + '_f_' + species_name
                eqn           = stoich * parse_expr(self.eqn_f_str)
                self.fluxes.update({flux_name: Flux(flux_name, species, eqn, self)})
            if self.eqn_r_str:
                flux_name     = self.name + '_r_' + species_name
                eqn           = -stoich * parse_expr(self.eqn_r_str)
                self.fluxes.update({flux_name: Flux(flux_name, species, eqn, self)})


class FluxContainer(ObjectContainer):
    def __init__(self):
        super().__init__(Flux)

        # self.properties_to_print = ['species_name', 'equation', 'sign', 'involved_species',
        #                      'involved_parameters', 'source_compartment',
        #                      'destination_compartment', 'ukeys', 'group']

        self.properties_to_print = ['species_name', 'equation']#, 'ukeys']#'source_compartment', 'destination_compartment', 'ukeys']

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
        # self.sym_list = [str(x) for x in self.equation.free_symbols]
        # self.lambda_eqn = sym.lambdify(self.sym_list, self.equation, modules=['sympy','numpy'])
        # self.involved_species = list(self.species_map.keys())
        # self.involved_parameters = list(self.param_map.keys())
        self._check_input_type_validity()
        self.check_validity()

        # Getting additional flux properties
        self._post_init_get_involved_species_parameters_compartments()
        self._post_init_get_lambda_equation()
        self._post_init_get_flux_topology()
        self._post_init_get_boundary_marker()

        # Get dolfin flux
        self._post_init_flux_to_dolfin()

        
        # self.get_boundary_marker()
        # self.get_flux_units()
        # self.get_is_linear()
        # self.get_is_linear_comp()
        # self.get_ukeys(solver_system)
        # self.get_integration_measure(cc, solver_system)

    def _post_init_get_involved_species_parameters_compartments(self):
        self.destination_compartment = self.destination_species.compartment
        
        # Get the subset of species/parameters/compartments that are relevant
        params_and_species = {str(x) for x in self.equation.free_symbols}
        all_params  = self.reaction.parameters
        all_species = self.reaction.species
        self.parameters    = {x: all_params[x] for x in params_and_species.intersection(all_params.keys())}
        self.species       = {x: all_species[x] for x in params_and_species.intersection(all_species.keys())}
        self.compartments  = self.reaction.compartments
        # print(self.compartments)
        # assert self.compartments == {x.compartment_name: x for x in self.species.values()}
    
    def _post_init_get_lambda_equation(self):
        self.lambda_equation = sym.lambdify(list(self.species.keys()) + list(self.parameters.keys()), 
                                              self.equation, modules=['sympy','numpy'])
    
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
        [2d] volume_to_volume:          BC of u ()
        [2d] volume_to_surface:         PDE of v
        [2d] surface_to_volume:         BC of u
        [3d] volume-surface_to_volume:  BC of u ()
        [3d] volume-volume_to_surface:  PDE of v ()
        """
        # 1 compartment flux
        if self.reaction.topology in ['volume', 'surface']:
            self.topology = self.reaction.topology
            source_compartments = {self.destination_compartment.name}
        # 2 or 3 compartment flux
        elif self.reaction.topology in ['volume_volume', 'volume_surface', 'volume_surface_volume']:
            source_compartments = set(self.compartments.keys()).difference({self.destination_compartment.name})
            
            if self.reaction.topology == 'volume_volume':
                self.topology = 'volume_to_volume'

            elif self.reaction.topology == 'volume_surface':
                if self.destination_compartment.is_volume_mesh:
                    self.topology = 'surface_to_volume'
                else:
                    self.topology = 'volume_to_surface'

            elif self.reaction.topology == 'volume_surface_volume':
                if self.destination_compartment.is_volume_mesh:
                    self.topology = 'volume-surface_to_volume'
                else:
                    self.topology = 'volume-volume_to_surface'
            
        self.source_compartments = {name: self.reaction.compartments[name] for name in source_compartments}

        # Based on topology we know if it is a boundary condition or RHS term
        if self.topology in ['volume', 'surface', 'volume_to_surface', 'volume-volume_to_surface']:
            self.is_boundary_condition = False
        elif self.topology in ['surface_to_volume', 'volume_to_volume', 'volume-surface_to_volume']:
            self.is_boundary_condition = True

    def _post_init_get_boundary_marker(self):
        # if self.is_boundary_condition:
        #     self.boundary_marker = self.sourcek
            
            
        # dim = self.flux_dimensionality
        # if dim[1] <= dim[0]:
        #     self.boundary_marker = None
        # elif dim[1] > dim[0]: # boundary flux
        #     self.boundary_marker = self.involved_compartments[self.source_compartment].first_index_marker
        print("FIXME")
        pass
        
    def _post_init_flux_to_dolfin(self):
        variables = {}

        for parameter in self.parameters.values():
            if parameter.is_time_dependent:
                variables[parameter.name] = parameter.dolfin_constant * parameter.unit
            else:
                variables[parameter.name] = parameter.value_unit
        
        for species in self.species.values():
            variables[species.name] = species.u['u'] * species.concentration_units
        
        self.equation_eval  = self.lambda_equation(**variables)
        self.evaluate_equation()
        
    def evaluate_equation(self):
        "Updates equation_value and equation_units"
        self.equation_value = self.equation_eval.magnitude
        self.equation_units = common.pint_unit_to_quantity(self.equation_eval.units)
    
    def get_additional_flux_properties(self, cc, solver_system):
        # get additional properties of the flux
        self.get_involved_species_parameters_compartment(cc)
        self.get_flux_dimensionality()
        self.get_boundary_marker()
        self.get_flux_units()
        self.get_is_linear()
        self.get_is_linear_comp()
        self.get_ukeys(solver_system)
        self.get_integration_measure(cc, solver_system)

    def get_involved_species_parameters_compartment(self, cc):
        sym_str_list = {str(x) for x in self.sym_list}
        self.involved_species = sym_str_list.intersection(self.species_map.keys())
        self.involved_species.add(self.species_name)
        self.involved_parameters = sym_str_list.intersection(self.param_map.keys())

        # truncate species_map and param_map so they only contain the species and parameters we need
        self.species_map = dict((k, self.species_map[k]) for k in self.involved_species if k in self.species_map)
        self.param_map = dict((k, self.param_map[k]) for k in self.involved_parameters if k in self.param_map)

        self.involved_compartments = dict([(sp.compartment.name, sp.compartment) for sp in self.species_map.values()])

        if self.explicit_restriction_to_domain:
            self.involved_compartments.update({self.explicit_restriction_to_domain: cc[self.explicit_restriction_to_domain]})
        if len(self.involved_compartments) not in (1,2):
            raise Exception("Number of compartments involved in a flux must be either one or two!")
    #def flux_to_dolfin(self):

    def get_flux_dimensionality(self):
        destination_compartment = self.species_map[self.species_name].compartment
        destination_dim = destination_compartment.dimensionality
        comp_names = set(self.involved_compartments.keys())
        comp_dims = set([comp.dimensionality for comp in self.involved_compartments.values()])
        comp_names.remove(destination_compartment.name)
        comp_dims.remove(destination_dim)

        if len(comp_names) == 0:
            self.flux_dimensionality = [destination_dim]*2
            self.source_compartment = destination_compartment.name
        else:
            source_dim = comp_dims.pop()
            self.flux_dimensionality = [source_dim, destination_dim]
            self.source_compartment = comp_names.pop()

        self.destination_compartment = destination_compartment.name

    def get_boundary_marker(self):
        dim = self.flux_dimensionality
        if dim[1] <= dim[0]:
            self.boundary_marker = None
        elif dim[1] > dim[0]: # boundary flux
            self.boundary_marker = self.involved_compartments[self.source_compartment].first_index_marker

    def get_flux_units(self):
        sp = self.species_map[self.species_name]
        compartment_units = sp.compartment.compartment_units
        # a boundary flux
        if (self.boundary_marker and self.flux_dimensionality[1]>self.flux_dimensionality[0]):
            self.flux_units = sp.concentration_units / compartment_units * sp.diffusion_units
        else:
            self.flux_units = sp.concentration_units / unit.s

    def get_is_linear(self):
        """
        For a given flux we want to know which terms are linear
        """
        is_linear_wrt = {}
        for sym_var in self.sym_list:
            var_name = str(sym_var)
            if var_name in self.involved_species:
                if sym.diff(self.equation, var_name , 2).is_zero:
                    is_linear_wrt[var_name] = True
                else:
                    is_linear_wrt[var_name] = False

        self.is_linear_wrt = is_linear_wrt

    def get_is_linear_comp(self):
        """
        Is the flux linear in terms of a compartment vector (e.g. dj/du['pm'])
        """
        is_linear_wrt_comp = {}
        umap = {}

        for var_name in self.sym_list:
            if var_name in self.involved_species:
                comp_name = self.species_map[var_name].compartment_name
                umap.update({var_name: 'u'+comp_name})

        new_eqn = self.equation.subs(umap)

        for comp_name in self.involved_compartments:
            if sym.diff(new_eqn, 'u'+comp_name, 2).is_zero:
                is_linear_wrt_comp[comp_name] = True
            else:
                is_linear_wrt_comp[comp_name] = False

        self.is_linear_wrt_comp = is_linear_wrt_comp

    def get_integration_measure(self, cc, solver_system):
        sp = self.species_map[self.species_name]
        flux_dim = self.flux_dimensionality
        min_dim = min(cc.get_property('dimensionality').values())
        max_dim = max(cc.get_property('dimensionality').values())

        # boundary flux
        if flux_dim[0] < flux_dim[1]:
            self.int_measure = sp.compartment.ds(self.boundary_marker)
        # volumetric flux (max dimension)
        elif flux_dim[0] == flux_dim[1] == max_dim:
            self.int_measure = sp.compartment.dx
        # volumetric flux (min dimension)
        elif flux_dim[1] == min_dim < max_dim:
            if solver_system.ignore_surface_diffusion:
                self.int_measure = sp.compartment.dP
            else:
                self.int_measure = sp.compartment.dx
        else:
            raise Exception("I'm not sure what integration measure to use on a flux with this dimensionality")

    def get_ukeys(self, solver_system):
        """
        Given the dimensionality of a flux (e.g. 2d surface to 3d vol) and the dimensionality
        of a species, determine which term of u should be used
        """
        self.ukeys = {}
        flux_vars = [str(x) for x in self.sym_list if str(x) in self.involved_species]
        for var_name in flux_vars:
            self.ukeys[var_name] = self.get_ukey(var_name, solver_system)

    def get_ukey(self, var_name, solver_system):
        sp = self.species_map[self.species_name]
        var = self.species_map[var_name]

        if solver_system.nonlinear_solver.method == 'newton':
            # if var.dimensionality > sp.dimensionality:
            #     return 'b'+sp.compartment_name
            # else:
            #     return 'u'

            # Testing volume interpolated functions
            if var.dimensionality > sp.dimensionality:
                return 'b_'+sp.compartment_name
            elif var.dimensionality < sp.dimensionality:
                return 'v_'+sp.compartment_name
            else:
                return 'u'

        elif solver_system.nonlinear_solver.method == 'IMEX':
            ## same compartment
            # dynamic lhs
            # if var.name == sp.name:
            #     if self.is_linear_wrt[sp.name]:
            #         return 't'
            #     else:
            #         return 'n'
            # static lhs
            if var.compartment_name == sp.compartment_name:
                if self.is_linear_wrt_comp[sp.compartment_name]:
                    return 't'
                else:
                    return 'n'
            ## different compartments
            # volume -> surface
            if var.dimensionality > sp.dimensionality:
                return 'b_'+sp.compartment_name
            # surface -> volume is covered by first if statement in get_ukey()

        raise Exception("Missing logic in get_ukey(); contact a developer...")

    # def flux_to_dolfin(self):
    #     value_dict = {}

    #     for var_name in [str(x) for x in self.sym_list]:
    #         if var_name in self.param_map.keys():
    #             var = self.param_map[var_name]
    #             if var.is_time_dependent:
    #                 value_dict[var_name] = var.dolfin_constant * var.unit
    #             else:
    #                 value_dict[var_name] = var.value_unit
    #         elif var_name in self.species_map.keys():
    #             var = self.species_map[var_name]
    #             ukey = self.ukeys[var_name]
    #             value_dict[var_name] = var.u[ukey]

    #             value_dict[var_name] *= var.concentration_units * 1

    #     eqn_eval = self.lambda_eqn(**value_dict)
    #     prod = eqn_eval.magnitude
    #     unit_prod = 1 * (1*eqn_eval.units).units
    #     #unit_prod = self.lambda_eqn(**unit_dict)
    #     #unit_prod = 1 * (1*unit_prod).units # trick to make object a "Quantity" class

    #     self.prod = prod
    #     self.unit_prod = unit_prod

class FormContainer:
    def __init__(self):
        self.form_list = []
    def add(self, new_form):
        self.form_list.append(new_form)
    def select_by(self, selection_key, value):
        return [f for f in self.form_list if getattr(f, selection_key)==value]
    def inspect(self, form_list=None):
        if not form_list:
            form_list = self.form_list

        for index, form in enumerate(form_list):
            Print("Form with index %d from form_list..." % index)
            if form.flux_name:
                Print("Flux name: %s" % form.flux_name)
            Print("Species name: %s" % form.species_name)
            Print("Form type: %s" % form.form_type)
            form.inspect()


class Form:
    def __init__(self, dolfin_form, species, form_type, flux_name=None):
        # form_type:
        # 'M': transient/mass form (holds time derivative)
        # 'D': diffusion form
        # 'R': domain reaction forms
        # 'B': boundary reaction forms

        self.dolfin_form = dolfin_form
        self.species = species
        self.species_name = species.name
        self.compartment_name = species.compartment_name
        self.form_type = form_type
        self.flux_name = flux_name

    def inspect(self):
        integrals = self.dolfin_form.integrals()
        for index, integral in enumerate(integrals):
            Print(str(integral) + "\n")
