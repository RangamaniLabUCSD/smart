""" 
Classes for parameters, species, compartments, reactions, fluxes, and forms
Model class contains functions to efficiently solve a system
"""

import re
from collections import Counter
from collections import OrderedDict as odict
from collections import defaultdict as ddict
from termcolor import colored
import pandas as pd
import dolfin as d
import mpi4py.MPI as pyMPI
import petsc4py.PETSc as PETSc
Print = PETSc.Sys.Print

import sympy
from sympy.parsing.sympy_parser import parse_expr
from sympy import Heaviside, lambdify
from sympy.utilities.iterables import flatten

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import cumtrapz

import pint
from pprint import pprint
from tabulate import tabulate
from copy import copy, deepcopy

import stubs.common as common
import stubs
from stubs import unit as ureg

comm = d.MPI.comm_world
rank = comm.rank
size = comm.size
root = 0


def color_print(full_text, color):
    if rank==root:
        split_text = [s for s in re.split('(\n)', full_text) if s] # colored doesn't like newline characters
        for text in split_text:
            if text == '\n':
                print()
            else:
                print(colored(text, color=color))


# ====================================================
# ====================================================
# Base Classes
# ====================================================
# ====================================================
class _ObjectContainer(object):
    """
    Parent class containing general methods used by all "containers"
    """
    def __init__(self, ObjectClass, df=None, Dict=None):
        self.Dict = odict()
        self.dtypes = {}
        self.ObjectClass = ObjectClass
        self.properties_to_print = [] # properties to print
        #        self.name_key = name_key
        if df is not None:
            self.add_pandas_dataframe(df)
        if Dict is not None:
            self.Dict = odict(Dict)
    def add_property(self, property_name, item):
        setattr(self, property_name, item)
    def add_property_to_all(self, property_name, item):
        for obj in self.Dict.values():
            setattr(obj, property_name, item)
    def add(self, item):
        self.Dict[item.name] = item
    def remove(self, name):
        self.Dict.pop(name)
    def add_pandas_dataframe(self, df):
        for _, row in df.iterrows():
            itemDict = row.to_dict()
            self.Dict[row.name] = self.ObjectClass(row.name, Dict=itemDict)
        self.dtypes.update(df.dtypes.to_dict())
    def get_property(self, property_name):
        # returns a dict of properties
        property_dict = {}
        for key, obj in self.Dict.items():
            property_dict[key] = getattr(obj, property_name)
        return property_dict

    def where_equals(self, property_name, value):
        """
        Links objects from ObjectContainer2 to ObjectContainer1 (the _ObjectContainer invoking
        this method).

        Parameters
        ----------
        property_name : str
            Name of property to check values of
        value : variable type
            Value to check against

        Example Usage
        -------------
        CD.where_equals('compartment_name', 'cyto')

        Returns
        -------
        objectList: list
            List of objects from _ObjectContainer that matches the criterion
        """
        objList = []
        for key, obj in self.Dict.items():
            if getattr(obj, property_name) == value:
                objList.append(obj)
        return objList

    def link_object(self, ObjectContainer2, property_name1, property_name2, linked_name, value_is_key=False):
        """
        Links objects from ObjectContainer2 to ObjectContainer1 (the _ObjectContainer invoking
        this method).

        Parameters
        ----------
        with_attribution : bool, Optional, default: True
            Set whether or not to display who the quote is from
        ObjectContainer2 : _ObjectContainer type
            _ObjectContainer with objects we are linking to
        property_name1 : str
            Name of property of object in ObjectContainer1 to match
        property_name2 : str
            Name of property of object in ObjectContainer2 to match
        linked_name : str
            Name of new property in ObjectContainer1 with linked object

        Example Usage
        -------------
        SD = {'key0': sd0, 'key1': sd1}; CD = {'key0': cd0, 'key1': cd1}
        sd0.compartment_name == 'cyto'
        cd0.name == 'cyto'
        >> SD.link_object(CD,'compartment_name','name','compartment')
        sd0.compartment == cd0

        Returns
        -------
        ObjectContainer1 : _ObjectContainer
            _ObjectContainer where each object has an added property linking to some
            object from ObjectContainer2
        """
        for _, obj1 in self.Dict.items():
            obj1_value = getattr(obj1, property_name1)
            # if type dict, then match values of entries with ObjectContainer2
            if type(obj1_value) == dict:
                newDict = odict()
                for key, value in obj1_value.items():
                    objList = ObjectContainer2.where_equals(property_name2, value)
                    if len(objList) != 1:
                        raise Exception("Property %s with value %s does not match %s (either none or more than one objects match)"
                                        % (property_name2, value, objList))
                    if value_is_key:
                        newDict.update({value: objList[0]})
                    else:
                        newDict.update({key: objList[0]})

                setattr(obj1, linked_name, newDict)
            #elif type(obj1_value) == list or type(obj1_value) == set:
            #    newList = []
            #    for value in obj1_value:
            #        objList = ObjectContainer2.where_equals(property_name2, value)
            #        if len(objList) != 1:
            #            raise Exception('Either none or more than one objects match this condition')
            #        newList.append(objList[0])
            #    setattr(obj1, linked_name, newList)
            elif type(obj1_value) == list or type(obj1_value) == set:
                newDict = odict()
                for value in obj1_value:
                    objList = ObjectContainer2.where_equals(property_name2, value)
                    if len(objList) != 1:
                        raise Exception("Property %s with value %s does not match %s (either none or more than one objects match)"
                                        % (property_name2, value, objList))
                    newDict.update({value: objList[0]})
                setattr(obj1, linked_name, newDict)
            # standard behavior
            else:
                objList = ObjectContainer2.where_equals(property_name2, obj1_value)
                if len(objList) != 1:
                    raise Exception("Property %s with value %s does not match %s (either none or more than one objects match)"
                                    % (property_name2, obj1_value, objList))
                setattr(obj1, linked_name, objList[0])

    def copy_linked_property(self, linked_name, linked_name_property, property_name):
        """
        Convenience function to copy a property from a linked object
        """
        for _, obj in self.Dict.items():
            linked_obj = getattr(obj, linked_name)
            setattr(obj, property_name, getattr(linked_obj, linked_name_property))


    def do_to_all(self, method_name, kwargs=None):
        for name, instance in self.Dict.items():
            if kwargs is None:
                getattr(instance, method_name)()
            else:
                getattr(instance, method_name)(**kwargs)

    def get_pandas_dataframe(self, properties_to_print=[], include_idx=True):
        df = pd.DataFrame()
        if include_idx:
            if properties_to_print and 'idx' not in properties_to_print:
                properties_to_print.insert(0, 'idx')
            for idx, (name, instance) in enumerate(self.Dict.items()):
                df = df.append(instance.get_pandas_series(properties_to_print=properties_to_print, idx=idx))
        else:
            for idx, (name, instance) in enumerate(self.Dict.items()):
                df = df.append(instance.get_pandas_series(properties_to_print=properties_to_print))
        # sometimes types are recast. change entries into their original types
        for dtypeName, dtype in self.dtypes.items():
            if dtypeName in df.columns:
                df = df.astype({dtypeName: dtype})

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

        with pd.option_context("max_colwidth", 1000):
            tex_str = df.to_latex(index=False, longtable=True, escape=escape)
            print(tex_str)

    def get_index(self, idx):
        """
        Get an element of the object container ordered dict by referencing its index
        """
        return list(self.Dict.values())[idx]

    def print(self, tablefmt='fancy_grid', properties_to_print=[]):
        if rank == root:
            if properties_to_print:
                if type(properties_to_print) != list: properties_to_print=[properties_to_print]
            elif hasattr(self, 'properties_to_print'):
                properties_to_print = self.properties_to_print
            df = self.get_pandas_dataframe(properties_to_print=properties_to_print)
            if properties_to_print:
                # make sure that all properties exist on the dataframe
                properties_to_print = list(set(properties_to_print).intersection(df.columns))
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
                keyList = list(self.Dict.keys())

            if properties_to_print:
                if type(properties_to_print) != list: properties_to_print=[properties_to_print]
            elif hasattr(self, 'properties_to_print'):
                properties_to_print = self.properties_to_print

            if print_all: properties_to_print = []
            for key in keyList:
                self.Dict[key].print(properties_to_print=properties_to_print)
        else:
            pass


class _ObjectInstance(object):
    """
    Parent class containing general methods used by all stubs
    "objects": i.e. parameters, species, compartments, reactions, fluxes, forms
    """
    def __init__(self, name, Dict=None):
        self.name = name
        if Dict:
            self.fromDict(Dict)
    def fromDict(self, Dict):
        for key, item in Dict.items():
            setattr(self, key, item)
    def combineDicts(self, dict1=None, dict2=None, new_dict_name=None):
        setattr(self, new_dict_name, getattr(self,dict1).update(getattr(self,dict2)))
    def assemble_units(self, value_name=None, unit_name='unit', assembled_name=None):
        """
        Simply multiplies a value by a unit (pint type) to create a pint "Quantity" object 
        """
        if not assembled_name:
            assembled_name = unit_name

        value = 1 if not value_name else getattr(self, value_name)
        #value = getattr(self, value_name)
        unit = getattr(self, unit_name)
        if type(unit) == str:
            unit = ureg(unit)
        setattr(self, assembled_name, value*unit)

    def get_pandas_series(self, properties_to_print=[], idx=None):
        if properties_to_print:
            dict_to_convert = odict({'idx': idx})
            dict_to_convert.update(odict([(key,val) for (key,val) in self.__dict__.items() if key in properties_to_print]))
        else:
            dict_to_convert = self.__dict__
        return pd.Series(dict_to_convert, name=self.name)
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

class ParameterContainer(_ObjectContainer):
    def __init__(self, df=None, Dict=None):
        super().__init__(Parameter, df, Dict)
        self.properties_to_print = ['name', 'value', 'unit', 'is_time_dependent', 'symExpr', 'notes', 'group']

class Parameter(_ObjectInstance):
    def __init__(self, name, Dict=None):
        super().__init__(name, Dict)
    def assembleTimeDependentParameters(self, zero_d=False):
        #TODO
        if not self.is_time_dependent:
            return
        # Parse the given string to create a sympy expression
        if self.symExpr or zero_d:
            self.symExpr = parse_expr(self.symExpr)
            Print("Creating dolfin object for time-dependent parameter %s" % self.name)
            self.dolfinConstant = d.Constant(self.value)
        if self.preintegrated_symExpr:
            self.preintegrated_symExpr = parse_expr(self.preintegrated_symExpr)
        # load in sampling data file
        if self.sampling_file:
            self.sampling_data = np.genfromtxt(self.sampling_file, dtype='float',
                                               delimiter=',')
            Print("Creating dolfin object for time-dependent parameter %s" % self.name)
            self.dolfinConstant = d.Constant(self.value)

            # preintegrate sampling data
            int_data = cumtrapz(self.sampling_data[:,1], x=self.sampling_data[:,0], initial=0)
            # concatenate time vector
            self.preint_sampling_data = common.np_smart_hstack(self.sampling_data[:,0], int_data)



class SpeciesContainer(_ObjectContainer):
    def __init__(self, df=None, Dict=None):
        super().__init__(Species, df, Dict)
        self.properties_to_print = ['name', 'compartment_name', 'compartment_index', 'concentration_units', 'D', 'initial_condition', 'group']

    def assemble_compartment_indices(self, RD, CD, settings):
        """
        Adds a column to the species dataframe which indicates the index of a species relative to its compartment
        """
        num_species_per_compartment = RD.get_species_compartment_counts(self, CD, settings)
        for compartment, num_species in num_species_per_compartment.items():
            idx = 0
            comp_species = [sp for sp in self.Dict.values() if sp.compartment_name==compartment]
            for sp in comp_species:
                if sp.is_in_a_reaction or sp.parent_species:
                    sp.compartment_index = idx
                    idx += 1
                else:
                    print('Warning: species %s is not used in any reactions!' % sp.name)


    def assemble_dolfin_functions(self, RD, CD, settings):
        """
        define dof/solution vectors (dolfin trialfunction, testfunction, and function types) based on number of species appearing in reactions
        IMPORTANT: this function will create additional species on boundaries in order to use operator-splitting later on
        e.g.
        A [cyto] + B [pm] <-> C [pm]
        Since the value of A is needed on the pm there will be a species A_b_pm which is just the values of A on the boundary pm
        """

        # functions to run beforehand as we need their results
        num_species_per_compartment = RD.get_species_compartment_counts(self, CD, settings)
        CD.get_min_max_dim()
        self.assemble_compartment_indices(RD, CD, settings)
        CD.add_property_to_all('is_in_a_reaction', False)
        CD.add_property_to_all('V', None)

        V, u, v = {}, {}, {}
        for compartment_name, num_species in num_species_per_compartment.items():
            compartmentDim = CD.Dict[compartment_name].dimensionality
            CD.Dict[compartment_name].num_species = num_species
            if rank==root:
                print('Compartment %s (dimension: %d) has %d species associated with it' %
                      (compartment_name, compartmentDim, num_species))

            # u is the actual function. t is for linearized versions. k is for picard iterations. n is for last time-step solution
            if num_species == 1:
                V[compartment_name] = d.FunctionSpace(CD.meshes[compartment_name], 'P', 1)
                u[compartment_name] = {'u': d.Function(V[compartment_name], name="concentration_u"), 't': d.TrialFunction(V[compartment_name]),
                'k': d.Function(V[compartment_name]), 'n': d.Function(V[compartment_name])}
                v[compartment_name] = d.TestFunction(V[compartment_name])
            else: # vector space
                V[compartment_name] = d.VectorFunctionSpace(CD.meshes[compartment_name], 'P', 1, dim=num_species)
                u[compartment_name] = {'u': d.Function(V[compartment_name], name="concentration_u"), 't': d.TrialFunctions(V[compartment_name]),
                'k': d.Function(V[compartment_name]), 'n': d.Function(V[compartment_name])}
                v[compartment_name] = d.TestFunctions(V[compartment_name])

        if not settings['add_boundary_species']: # if the setting is true sub_species will be added
            # now we create boundary functions, which are defined on the function spaces of the surrounding mesh
            V['boundary'] = {}
            for compartment_name, num_species in num_species_per_compartment.items():
                compartmentDim = CD.Dict[compartment_name].dimensionality
                if compartmentDim == CD.max_dim: # mesh may have boundaries
                    V['boundary'][compartment_name] = {}
                    for boundary_name, boundary_mesh in CD.meshes.items():
                        if compartment_name != boundary_name:
                            if num_species == 1:
                                boundaryV = d.FunctionSpace(CD.meshes[boundary_name], 'P', 1)
                            else:
                                boundaryV = d.VectorFunctionSpace(CD.meshes[boundary_name], 'P', 1, dim=num_species)
                            V['boundary'][compartment_name].update({boundary_name: boundaryV})
                            u[compartment_name]['b'+boundary_name] = d.Function(boundaryV, name="concentration_ub")

        # associate indexed functions with dataframe
        for key, sp in self.Dict.items():
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
                        sp.u.update({key: u[sp.compartment_name][key][sp.compartment_index]})
                        sp.v = v[sp.compartment_name][sp.compartment_index]

        # # associate function spaces with dataframe
        for key, comp in CD.Dict.items():
            if comp.is_in_a_reaction:
                comp.V = V[comp.name]

        self.u = u
        self.v = v
        self.V = V

    def assign_initial_conditions(self):
        keys = ['k', 'n', 'u']
        for sp in self.Dict.values():
            comp_name = sp.compartment_name
            for key in keys:
                # stubs.data_manipulation.dolfinSetFunctionValues(self.u[comp_name][key], sp.initial_condition,
                #                                           self.V[comp_name], sp.compartment_index)
                stubs.data_manipulation.dolfinSetFunctionValues(self.u[comp_name][key], sp.initial_condition,
                                                                sp.compartment_index)
            #self.u[comp_name]['u'].assign(self.u[comp_name]['n'])
            if rank==root: print("Assigned initial condition for species %s" % sp.name)

        # add boundary values
        for comp_name in self.u.keys():
            for ukey in self.u[comp_name].keys():
                if 'b' in key[0]:
                    self.u[comp_name][ukey].interpolate(self.u[comp_name]['u'])




class Species(_ObjectInstance):
    def __init__(self, name, Dict=None):
        super().__init__(name, Dict)
        self.sub_species = {} # additional compartments this species may live in in addition to its primary one
        self.is_in_a_reaction = False
        self.is_an_added_species = False
        self.parent_species = None
        self.dof_map = {}


class CompartmentContainer(_ObjectContainer):
    def __init__(self, df=None, Dict=None):
        super().__init__(Compartment, df, Dict)
        self.properties_to_print = ['name', 'dimensionality', 'num_species', 'num_vertices', 'cell_marker', 'is_in_a_reaction', 'nvolume']
        self.meshes = {}
        self.vertex_mappings = {} # from submesh -> parent indices
    def load_mesh(self, mesh_key, mesh_str):
        self.meshes[mesh_key] = d.Mesh(mesh_str)
    def extract_submeshes(self, main_mesh_str, save_to_file):
        main_mesh = self.Dict[main_mesh_str]
        surfaceDim = main_mesh.dimensionality - 1

        self.Dict[main_mesh_str].mesh = self.meshes[main_mesh_str]

        vmesh = self.meshes[main_mesh_str]
        bmesh = d.BoundaryMesh(vmesh, "exterior")

        # Very odd behavior - when bmesh.entity_map() is called together with .array() it will return garbage values. We
        # should only call entity_map once to avoid this
        emap_0 = bmesh.entity_map(0)
        bmesh_emap_0 = deepcopy(emap_0.array())
        emap_2 = bmesh.entity_map(2)
        bmesh_emap_2 = deepcopy(emap_2.array())
        vmf = d.MeshFunction("size_t", vmesh, surfaceDim, vmesh.domains())
        bmf = d.MeshFunction("size_t", bmesh, surfaceDim)
        for idx, facet in enumerate(d.entities(bmesh,surfaceDim)): # iterate through faces of bmesh
            #vmesh_idx = bmesh.entity_map(surfaceDim)[idx] # get the index of the face on vmesh corresponding to this face on bmesh
            vmesh_idx = bmesh_emap_2[idx] # get the index of the face on vmesh corresponding to this face on bmesh
            vmesh_boundarynumber = vmf.array()[vmesh_idx] # get the value of the mesh function at this face
            bmf.array()[idx] = vmesh_boundarynumber # set the value of the boundary mesh function to be the same value


        # Loop through compartments
        for key, obj in self.Dict.items():
            # FEniCS doesn't allow parallelization of SubMeshes. We need
            # SubMeshes because one boundary will often have multiple domains of
            # interest with different species (e.g., PM, ER). By exporting the
            # submeshes in serial we can reload them back in in parallel.
            if key!=main_mesh_str and obj.dimensionality==surfaceDim:
                # TODO: fix this
                if size > 1: # if we are running in parallel
                    Print("CPU %d: Loading submesh for %s from file" % (rank, key))
                    submesh = d.Mesh(d.MPI.comm_self, 'submeshes/submesh_' + obj.name + '_' + str(obj.cell_marker) + '.xml')
                    self.meshes[key] = submesh
                    obj.mesh = submesh
                else:
                    Print("Saving submeshes %s for use in parallel" % key)
                    submesh = d.SubMesh(bmesh, bmf, obj.cell_marker)
                    self.vertex_mappings[key] = submesh.data().array("parent_vertex_indices", 0)
                    self.meshes[key] = submesh
                    obj.mesh = submesh
                    if save_to_file:
                        save_str = 'submeshes/submesh_' + obj.name + '_' + str(obj.cell_marker) + '.xml'
                        d.File(save_str) << submesh
            # integration measures
            if obj.dimensionality==main_mesh.dimensionality:
                obj.ds = d.Measure('ds', domain=obj.mesh, subdomain_data=vmf, metadata={'quadrature_degree': 3})
                obj.dP = None
            elif obj.dimensionality<main_mesh.dimensionality:
                obj.dP = d.Measure('dP', domain=obj.mesh)
                obj.ds = None
            else:
                raise Exception("main_mesh is not a maximum dimension compartment")
            obj.dx = d.Measure('dx', domain=obj.mesh, metadata={'quadrature_degree': 3})

        # Get # of vertices
        for key, mesh in self.meshes.items():
            num_vertices = mesh.num_vertices()
            print('CPU %d: My partition of mesh %s has %d vertices' % (rank, key, num_vertices))
            self.Dict[key].num_vertices = num_vertices

        self.vmf = vmf
        self.bmesh = bmesh
        self.bmesh_emap_0 = bmesh_emap_0
        self.bmesh_emap_2 = bmesh_emap_2
        self.bmf = bmf

        # If we were running in serial to generate submeshes, exit here and
        # restart in parallel
        if save_to_file and size==1:
            Print("If run in serial, submeshes were saved to file. Run again"\
                  "in parallel.")
            exit()



    def compute_scaling_factors(self):
        for comp in self.Dict.values():
            if not hasattr(comp, 'nvolume'):
                comp.compute_nvolume()

        for key, obj in self.Dict.items():
            obj.scale_to = {}
            for key2, obj2 in self.Dict.items():
                if key != key2:
                    obj.scale_to.update({key2: ureg(obj.nvolume) / ureg(obj2.nvolume)})
    def get_min_max_dim(self):
        comp_dims = [comp.dimensionality for comp in self.Dict.values()]
        self.min_dim = min(comp_dims)
        self.max_dim = max(comp_dims)



class Compartment(_ObjectInstance):
    def __init__(self, name, Dict=None):
        super().__init__(name, Dict)
    def compute_nvolume(self):
        self.nvolume = d.assemble(d.Constant(1.0)*self.dx) * self.compartment_units ** self.dimensionality



class ReactionContainer(_ObjectContainer):
    def __init__(self, df=None, Dict=None):
        super().__init__(Reaction, df, Dict)
        #self.properties_to_print = ['name', 'LHS', 'RHS', 'eqn_f', 'eqn_r', 'paramDict', 'reaction_type', 'explicit_restriction_to_domain', 'group']
        self.properties_to_print = ['name', 'LHS', 'RHS', 'eqn_f']#, 'eqn_r']

    def get_species_compartment_counts(self, SD, CD, settings):
        self.do_to_all('get_involved_species_and_compartments', {"SD": SD, "CD": CD})
        all_involved_species = set([sp for species_set in [rxn.involved_species_link.values() for rxn in self.Dict.values()] for sp in species_set])
        for sp_name, sp in SD.Dict.items():
            if sp in all_involved_species:
                sp.is_in_a_reaction = True

        compartment_counts = [sp.compartment_name for sp in all_involved_species]


        if settings['add_boundary_species']:
            ### additional boundary functions
            # get volumetric species which should also be defined on their boundaries
            sub_species_to_add = []
            for rxn in self.Dict.values():
                involved_compartments = [CD.Dict[comp_name] for comp_name in rxn.involved_compartments]
                rxn_min_dim = min([comp.dimensionality for comp in involved_compartments])
                rxn_max_dim = min([comp.dimensionality for comp in involved_compartments])
                for sp_name in rxn.involved_species:
                    sp = SD.Dict[sp_name]
                    if sp.dimensionality > rxn_min_dim: # species is involved in a boundary reaction
                        for comp in involved_compartments:
                            if comp.name != sp.compartment_name:
                                sub_species_to_add.append((sp_name, comp.name))
                                #sp.sub_species.update({comp.name: None})

            # Create a new species on boundaries
            sub_sp_list = []
            for sp_name, comp_name in set(sub_species_to_add):
                sub_sp_name = sp_name+'_sub_'+comp_name
                compartment_counts.append(comp_name)
                if sub_sp_name not in SD.Dict.keys():
                    color_print('\nSpecies %s will have a new function defined on compartment %s with name: %s\n'
                        % (sp_name, comp_name, sub_sp_name), 'blue')

                    sub_sp = copy(SD.Dict[sp_name])
                    sub_sp.is_an_added_species = True
                    sub_sp.name = sub_sp_name
                    sub_sp.compartment_name = comp_name
                    sub_sp.compartment = CD.Dict[comp_name]
                    sub_sp.is_in_a_reaction = True
                    sub_sp.sub_species = {}
                    sub_sp.parent_species = sp_name
                    sub_sp_list.append(sub_sp)

            #if sub_sp_name not in SD.Dict.keys():
            for sub_sp in sub_sp_list:
                SD.Dict[sub_sp.name] = sub_sp
                SD.Dict[sub_sp.parent_species].sub_species.update({sub_sp.compartment_name: sub_sp})


        return Counter(compartment_counts)

    # def replace_sub_species_in_reactions(self, SD):
    #     """
    #     New species may be created to live on boundaries
    #     TODO: this may cause issues if new properties are added to SpeciesContainer
    #     """
    #     sp_to_replace = []
    #     for rxn in self.Dict.values():
    #         for sp_name, sp in rxn.involved_species_link.items():
    #             if set(sp.sub_species.keys()).intersection(rxn.involved_compartments):
    #             #if sp.sub_species:
    #                 print(sp.sub_species.items())
    #                 for sub_comp, sub_sp in sp.sub_species.items():
    #                     sub_sp_name = sub_sp.name

    #                     rxn.LHS = [sub_sp_name if x==sp_name else x for x in rxn.LHS]
    #                     rxn.RHS = [sub_sp_name if x==sp_name else x for x in rxn.RHS]
    #                     rxn.eqn_f = rxn.eqn_f.subs({sp_name: sub_sp_name})
    #                     rxn.eqn_r = rxn.eqn_r.subs({sp_name: sub_sp_name})

    #                     sp_to_replace.append((sp_name, sub_sp_name, sub_sp))

    #                     print('Species %s replaced with %s in reaction %s!!!' % (sp_name, sub_sp_name, rxn.name))

    #                 rxn.name = rxn.name + ' [modified]'

    #         for tup in sp_to_replace:
    #             sp_name, sub_sp_name, sub_sp = tup
    #             rxn.involved_species.remove(sp_name)
    #             rxn.involved_species.add(sub_sp_name)
    #             rxn.involved_species_link.pop(sp_name, None)
    #             rxn.involved_species_link.update({sub_sp_name: sub_sp})



    def reaction_to_fluxes(self):
        self.do_to_all('reaction_to_fluxes')
        fluxList = []
        for rxn in self.Dict.values():
            for f in rxn.fluxList:
                fluxList.append(f)
        self.fluxList = fluxList
    def get_flux_container(self):
        return FluxContainer(Dict=odict([(f.flux_name, f) for f in self.fluxList]))


class Reaction(_ObjectInstance):
    def __init__(self, name, Dict=None, eqn_f_str=None, eqn_r_str=None,
                 explicit_restriction_to_domain=False, speciesDict={}, track_value=False):
        if eqn_f_str:
            print("Reaction %s: using the specified equation for the forward flux: %s" % (name, eqn_f_str))
            self.eqn_f = parse_expr(eqn_f_str)
        if eqn_r_str:
            print("Reaction %s: using the specified equation for the reverse flux: %s" % (name, eqn_r_str))
            self.eqn_r = parse_expr(eqn_r_str)
        self.explicit_restriction_to_domain = explicit_restriction_to_domain
        self.track_value = track_value
        super().__init__(name, Dict)

    def initialize_flux_equations_for_known_reactions(self, reaction_database={}):
        """
        Generates unsigned forward/reverse flux equations for common/known reactions
        """
        if self.reaction_type == 'mass_action':
            rxnSymStr = self.paramDict['on']
            for sp_name in self.LHS:
                rxnSymStr += '*' + sp_name
            self.eqn_f = parse_expr(rxnSymStr)

            rxnSymStr = self.paramDict['off']
            for sp_name in self.RHS:
                rxnSymStr += '*' + sp_name
            self.eqn_r = parse_expr(rxnSymStr)

        elif self.reaction_type == 'mass_action_forward':
            rxnSymStr = self.paramDict['on']
            for sp_name in self.LHS:
                rxnSymStr += '*' + sp_name
            self.eqn_f = parse_expr(rxnSymStr)

        elif self.reaction_type in reaction_database.keys():
            self.custom_reaction(reaction_database[self.reaction_type])

        else:
            raise Exception("Reaction %s does not seem to have an associated equation" % self.name)


    def custom_reaction(self, symStr):
        rxnExpr = parse_expr(symStr)
        rxnExpr = rxnExpr.subs(self.paramDict)
        rxnExpr = rxnExpr.subs(self.speciesDict)
        self.eqn_f = rxnExpr

    def get_involved_species_and_compartments(self, SD=None, CD=None):
        # used to get number of active species in each compartment
        self.involved_species = set(self.LHS + self.RHS)
        for eqn in ['eqn_r', 'eqn_f']:
            if hasattr(self, eqn):
                varSet = {str(x) for x in self.eqn_f.free_symbols}
                spSet = varSet.intersection(SD.Dict.keys())
                self.involved_species = self.involved_species.union(spSet)

        self.involved_compartments = dict(set([(SD.Dict[sp_name].compartment_name, SD.Dict[sp_name].compartment) for sp_name in self.involved_species]))
        if self.explicit_restriction_to_domain:
            self.involved_compartments.update({self.explicit_restriction_to_domain: CD.Dict[self.explicit_restriction_to_domain]})

        if len(self.involved_compartments) not in (1,2):
            raise Exception("Number of compartments involved in a flux must be either one or two!")

    def reaction_to_fluxes(self):
        self.fluxList = []
        all_species = self.LHS + self.RHS
        unique_species = set(all_species)
        for species_name in unique_species:
            stoich = max([self.LHS.count(species_name), self.RHS.count(species_name)])
            #all_species.count(species_name)

            track = True if species_name == self.track_value else False

            if hasattr(self, 'eqn_f'):
                flux_name = self.name + ' (f) [' + species_name + ']'
                sign = -1 if species_name in self.LHS else 1
                signed_stoich = sign*stoich
                self.fluxList.append(Flux(flux_name, species_name, self.eqn_f, signed_stoich, self.involved_species_link,
                                          self.paramDictValues, self.group, self.explicit_restriction_to_domain, track))
            if hasattr(self, 'eqn_r'):
                flux_name = self.name + ' (r) [' + species_name + ']'
                sign = 1 if species_name in self.LHS else -1
                signed_stoich = sign*stoich
                self.fluxList.append(Flux(flux_name, species_name, self.eqn_r, signed_stoich, self.involved_species_link,
                                          self.paramDictValues, self.group, self.explicit_restriction_to_domain, track))



class FluxContainer(_ObjectContainer):
    def __init__(self, df=None, Dict=None):
        super().__init__(Flux, df, Dict)
        # self.properties_to_print = ['species_name', 'symEqn', 'sign', 'involved_species',
        #                      'involved_parameters', 'source_compartment',
        #                      'destination_compartment', 'ukeys', 'group']

        self.properties_to_print = ['species_name', 'symEqn', 'signed_stoich', 'ukeys']#'source_compartment', 'destination_compartment', 'ukeys']
    def check_and_replace_sub_species(self, SD, CD, config):
        fluxes_to_remove = []
        for flux_name, f in self.Dict.items():
            tagged_for_removal = False
            for sp_name, sp in f.spDict.items():
                if sp.sub_species and (f.destination_compartment in sp.sub_species.keys()
                                     or f.source_compartment in sp.sub_species.keys()):
                    tagged_for_removal = True
                    print("flux %s tagged for removal" % flux_name)
                    #for sub_sp_name in sp.sub_species.keys():
                    for sub_sp in sp.sub_species.values():
                        if sub_sp.compartment_name in f.involved_compartments:
                            f.symEqn = f.symEqn.subs({sp_name: sub_sp.name})
                            print("subbed %s for %s" % (sp_name, sub_sp.name))

            if tagged_for_removal:
                fluxes_to_remove.append(f)
                tagged_for_removal = False

        new_flux_list = []
        for f in fluxes_to_remove:
            #if SD.Dict[f.species_name].compartment_name == f.source_compartment:
            new_flux_name = f.flux_name + ' [sub**]'
            involved_species = [str(x) for x in f.symEqn.free_symbols if str(x) in SD.Dict.keys()]
            if not SD.Dict[f.species_name].sub_species:
                new_species_name = f.species_name
            else:
                new_species_name = SD.Dict[f.species_name].sub_species[f.source_compartment].name

            involved_species += [new_species_name] # add the flux species
            species_w_parent = [SD.Dict[x] for x in involved_species if SD.Dict[x].parent_species]
            #if f.species_name not in parent_species:

            print("symEqn")
            print(f.symEqn)
            print("free symbols = ")
            print([str(x) for x in f.symEqn.free_symbols])
            print("involved species = ")
            print(involved_species)
            spDict = {}
            for sp_name in involved_species:
                spDict.update({sp_name: SD.Dict[sp_name]})

            new_flux = Flux(new_flux_name, new_species_name, f.symEqn, f.signed_stoich,
                            spDict, f.paramDict, f.group, f.explicit_restriction_to_domain, f.track_value)
            new_flux.get_additional_flux_properties(CD, config)

            # get length scale factor
            comp1 = SD.Dict[species_w_parent[0].parent_species].compartment
            comp2 = species_w_parent[0].compartment_name
            print(comp1.name)
            #print(comp2)
            length_scale_factor = comp1.scale_to[comp2]
            print("computed length_scale_factor")
            setattr(new_flux, 'length_scale_factor', length_scale_factor)

            new_flux_list.append((new_flux_name, new_flux))
            #else:
            #    new_species_name =
            #    print("species name, source compartment: %s, %s" % (f.species_name, f.source_compartment))

        for flux_rm in fluxes_to_remove:
            Print('removing flux %s' %  flux_rm.flux_name)
            self.Dict.pop(flux_rm.flux_name)

        for (new_flux_name, new_flux) in new_flux_list:
            Print('adding flux %s' % new_flux_name)
            self.Dict.update({new_flux_name: new_flux})

class Flux(_ObjectInstance):
    def __init__(self, flux_name, species_name, symEqn, signed_stoich,
                 spDict, paramDict, group, explicit_restriction_to_domain=None, track_value=False):
        super().__init__(flux_name)

        self.flux_name = flux_name
        self.species_name = species_name
        self.symEqn = symEqn
        self.signed_stoich = signed_stoich
        self.spDict = spDict
        self.paramDict = paramDict
        self.group = group
        self.explicit_restriction_to_domain = explicit_restriction_to_domain
        self.track_value = track_value
        self.tracked_values = []

        self.symList = [str(x) for x in symEqn.free_symbols]
        self.lambdaEqn = sympy.lambdify(self.symList, self.symEqn, modules=['sympy','numpy'])
        self.involved_species = list(spDict.keys())
        self.involved_parameters = list(paramDict.keys())


    def get_additional_flux_properties(self, CD, config):
        # get additional properties of the flux
        self.get_involved_species_parameters_compartment(CD)
        self.get_flux_dimensionality()
        self.get_boundary_marker()
        self.get_flux_units()
        self.get_is_linear()
        self.get_is_linear_comp()
        self.get_ukeys(config)
        self.get_integration_measure(CD, config)

    def get_involved_species_parameters_compartment(self, CD):
        symStrList = {str(x) for x in self.symList}
        self.involved_species = symStrList.intersection(self.spDict.keys())
        self.involved_species.add(self.species_name)
        self.involved_parameters = symStrList.intersection(self.paramDict.keys())

        # truncate spDict and paramDict so they only contain the species and parameters we need
        self.spDict = dict((k, self.spDict[k]) for k in self.involved_species if k in self.spDict)
        self.paramDict = dict((k, self.paramDict[k]) for k in self.involved_parameters if k in self.paramDict)

        #self.involved_compartments = set([sp.compartment for sp in self.spDict.values()])
        self.involved_compartments = dict([(sp.compartment.name, sp.compartment) for sp in self.spDict.values()])

        if self.explicit_restriction_to_domain:
            self.involved_compartments.update({self.explicit_restriction_to_domain: CD.Dict[self.explicit_restriction_to_domain]})
        if len(self.involved_compartments) not in (1,2):
            raise Exception("Number of compartments involved in a flux must be either one or two!")
    #def flux_to_dolfin(self):

    def get_flux_dimensionality(self):
        destination_compartment = self.spDict[self.species_name].compartment
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
            self.boundary_marker = self.involved_compartments[self.source_compartment].cell_marker

    def get_flux_units(self):
        sp = self.spDict[self.species_name]
        compartment_units = sp.compartment.compartment_units
        # a boundary flux
        if (self.boundary_marker and self.flux_dimensionality[1]>self.flux_dimensionality[0]) or sp.parent_species:
            self.flux_units = sp.concentration_units / compartment_units * sp.D_units
        else:
            self.flux_units = sp.concentration_units / ureg.s

    def get_is_linear(self):
        """
        For a given flux we want to know which terms are linear
        """
        is_linear_wrt = {}
        for symVar in self.symList:
            varName = str(symVar)
            if varName in self.involved_species:
                if sympy.diff(self.symEqn, varName , 2).is_zero:
                    is_linear_wrt[varName] = True
                else:
                    is_linear_wrt[varName] = False

        self.is_linear_wrt = is_linear_wrt

    def get_is_linear_comp(self):
        """
        Is the flux linear in terms of a compartment vector (e.g. dj/du['pm'])
        """
        is_linear_wrt_comp = {}
        umap = {}

        for varName in self.symList:
            if varName in self.involved_species:
                compName = self.spDict[varName].compartment_name
                umap.update({varName: 'u'+compName})

        newEqn = self.symEqn.subs(umap)

        for compName in self.involved_compartments:
            if sympy.diff(newEqn, 'u'+compName, 2).is_zero:
                is_linear_wrt_comp[compName] = True
            else:
                is_linear_wrt_comp[compName] = False

        self.is_linear_wrt_comp = is_linear_wrt_comp

    def get_integration_measure(self, CD, config):
        sp = self.spDict[self.species_name]
        flux_dim = self.flux_dimensionality
        min_dim = min(CD.get_property('dimensionality').values())
        max_dim = max(CD.get_property('dimensionality').values())

        # boundary flux
        if flux_dim[0] < flux_dim[1]:
            self.int_measure = sp.compartment.ds(self.boundary_marker)
        # volumetric flux (max dimension)
        elif flux_dim[0] == flux_dim[1] == max_dim:
            self.int_measure = sp.compartment.dx
        # volumetric flux (min dimension)
        elif flux_dim[1] == min_dim < max_dim:
            if config.settings['ignore_surface_diffusion']:
                self.int_measure = sp.compartment.dP
            else:
                self.int_measure = sp.compartment.dx
        else:
            raise Exception("I'm not sure what integration measure to use on a flux with this dimensionality")

    def get_ukeys(self, config):
        """
        Given the dimensionality of a flux (e.g. 2d surface to 3d vol) and the dimensionality
        of a species, determine which term of u should be used
        """
        self.ukeys = {}
        flux_vars = [str(x) for x in self.symList if str(x) in self.involved_species]
        for var_name in flux_vars:
            self.ukeys[var_name] = self.get_ukey(var_name, config)

    def get_ukey(self, var_name, config):
        sp = self.spDict[self.species_name]
        var = self.spDict[var_name]

        # boundary fluxes (surface -> volume)
        #if var.dimensionality < sp.dimensionality:
        if var.dimensionality < sp.compartment.dimensionality:
            return 'u' # always true if operator splitting to decouple compartments

        if sp.name == var.parent_species:
            return 'u'

        if config.solver['nonlinear'] == 'picard':# or 'IMEX':
            # volume -> surface
            if var.dimensionality > sp.dimensionality and config.settings['add_boundary_species']:
                if self.is_linear_wrt_comp[var.compartment_name]:
                    return 'bt'
                else:
                    return 'bk'

            elif var.dimensionality > sp.dimensionality:
                return 'b'+sp.compartment_name

            # volumetric fluxes
            elif var.compartment_name == self.destination_compartment:
                if self.is_linear_wrt_comp[var.compartment_name]:
                    return 't'
                #dynamic LHS
                elif var.name == sp.name and self.is_linear_wrt[sp.name]:
                    return 't'
                else:
                    return 'k'

        elif config.solver['nonlinear'] == 'newton':
            if var.dimensionality > sp.dimensionality:
                return 'b'+sp.compartment_name
            else:
                return 'u'

        elif config.solver['nonlinear'] == 'IMEX':
            ## same compartment
            # dynamic LHS
            # if var.name == sp.name:
            #     if self.is_linear_wrt[sp.name]:
            #         return 't'
            #     else:
            #         return 'n'
            # static LHS
            if var.compartment_name == sp.compartment_name:
                if self.is_linear_wrt_comp[sp.compartment_name]:
                    return 't'
                else:
                    return 'n'
            ## different compartments
            # volume -> surface
            if var.dimensionality > sp.dimensionality:
                return 'b'+sp.compartment_name
            # surface -> volume is covered by first if statement in get_ukey()




            # if sp.dimensionality == 3: #TODO fix this
            #     if var.compartment_name == sp.compartment_name and self.is_linear_wrt_comp[var.compartment_name]:
            #         return 't'
            #     else:
            #         if var.name == sp.name and self.is_linear_wrt[sp.name]:
            #             return 't'
            #         else:
            #             return 'k'
            # # volume -> surface
            # elif var.dimensionality > sp.dimensionality:
            #     return 'b'+sp.compartment_name
            # else:
            #     if self.is_linear_wrt_comp[var.compartment_name]:
            #         return 't'
            #     else:
            #         return 'k'

        # elif config.solver['nonlinear'] == 'IMEX':
        #     if
        #     return 'n'

        raise Exception("If you made it to this far in get_ukey() I missed some logic...")




    def flux_to_dolfin(self, config):
        value_dict = {}

        for var_name in [str(x) for x in self.symList]:
            if var_name in self.paramDict.keys():
                var = self.paramDict[var_name]
                if var.is_time_dependent:
                    value_dict[var_name] = var.dolfinConstant * var.unit
                else:
                    value_dict[var_name] = var.value_unit
            elif var_name in self.spDict.keys():
                var = self.spDict[var_name]
                ukey = self.ukeys[var_name]
                if ukey[0] == 'b' and config.settings['add_boundary_species']:
                    if not var.parent_species and config.settings['add_boundary_species']:
                        sub_species = var.sub_species[self.destination_compartment]
                        value_dict[var_name] = sub_species.u[ukey[1]]
                        Print("Species %s substituted for %s in flux %s" % (var_name, sub_species.name, self.name))
                    else:
                        value_dict[var_name] = var.u[ukey[1]]
                else:
                    value_dict[var_name] = var.u[ukey]

                value_dict[var_name] *= var.concentration_units * 1

        eqn_eval = self.lambdaEqn(**value_dict)
        prod = eqn_eval.magnitude
        unit_prod = 1 * (1*eqn_eval.units).units
        #unit_prod = self.lambdaEqn(**unit_dict)
        #unit_prod = 1 * (1*unit_prod).units # trick to make object a "Quantity" class

        self.prod = prod
        self.unit_prod = unit_prod

class FormContainer(object):
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


class Form(object):
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

# ==============================================================================
# ==============================================================================
# Model class. Consists of parameters, species, etc. and is used for simulation
# ==============================================================================
# ==============================================================================

class Model(object):
    def __init__(self, PD, SD, CD, RD, FD, config):
        self.PD = PD
        self.SD = SD
        self.CD = CD
        self.RD = RD
        self.FD = FD
        self.config = config

        if hasattr(SD, 'u'):
            self.u = SD.u
        if hasattr(SD, 'v'):
            self.v = SD.v
        if hasattr(SD, 'V'):
            self.V = SD.V

        self.params = ddict(list)

        self.idx = 0
        self.NLidx = 0 # nonlinear iterations
        self.t = 0.0
        self.dt = config.solver['initial_dt']
        self.T = d.Constant(self.t)
        self.dT = d.Constant(self.dt)
        self.t_final = config.solver['T']
        self.linear_iterations = None
        self.reset_dt = False

        self.timers = {}
        self.timings = ddict(list)

        self.Forms = FormContainer()
        self.a = {}
        self.L = {}
        self.F = {}
        self.nonlinear_solver = {}
        self.linear_solver = {}
        self.scipy_odes = {}

        self.data = stubs.data_manipulation.Data(self)
#===============================================================================
#===============================================================================
# PROBLEM SETUP
#===============================================================================
#===============================================================================
    def assemble_reactive_fluxes(self):
        """
        Creates the actual dolfin objects for each flux. Checks units for consistency
        """
        for j in self.FD.Dict.values():
            total_scaling = 1.0 # all adjustments needed to get congruent units
            sp = j.spDict[j.species_name]
            prod = j.prod
            unit_prod = j.unit_prod
            # first, check unit consistency
            if (unit_prod/j.flux_units).dimensionless: # this is the only case in 0d case
                setattr(j, 'scale_factor', 1*ureg.dimensionless)
                pass
            elif self.zero_d == True:
                raise Exception("Units should never be inconsistent in 0d case.")
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

                Print(('\nThe flux, %s, from compartment %s to %s, has units ' %
                       (j.flux_name, j.source_compartment, j.destination_compartment) + colored(unit_prod, "red") +
                       "...the desired units for this flux are " + colored(j.flux_units, "cyan")))

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
            if not self.zero_d:
                if j.flux_dimensionality[0] < j.flux_dimensionality[1]:
                    form_key = 'B'
                else:
                    form_key = 'R'
                prod = prod*sp.v*j.int_measure

                setattr(j, 'dolfin_flux', prod)

                BRform = -prod # by convention, terms are all defined as if they were on the LHS of the equation e.g. F(u;v)=0
                self.Forms.add(Form(BRform, sp, form_key, flux_name=j.name))


    def assemble_diffusive_fluxes(self):
        min_dim = min(self.CD.get_property('dimensionality').values())
        max_dim = max(self.CD.get_property('dimensionality').values())
        dT = self.dT

        for sp_name, sp in self.SD.Dict.items():
            if sp.is_in_a_reaction:
                if self.config.solver['nonlinear'] in ['picard', 'IMEX']:
                    u = sp.u['t']
                elif self.config.solver['nonlinear'] == 'newton':
                    u = sp.u['u']
                un = sp.u['n']
                v = sp.v
                D = sp.D

                if sp.dimensionality == max_dim and not sp.parent_species:
                    #or not self.config.settings['ignore_surface_diffusion']:
                    dx = sp.compartment.dx
                    Dform = D*d.inner(d.grad(u), d.grad(v)) * dx
                    self.Forms.add(Form(Dform, sp, 'D'))
                elif sp.dimensionality < max_dim or sp.parent_species:
                    if self.config.settings['ignore_surface_diffusion']:
                        dx=sp.compartment.dP
                    else:
                        dx = sp.compartment.dx
                        Dform = D*d.inner(d.grad(u), d.grad(v)) * dx
                        self.Forms.add(Form(Dform, sp, 'D'))

                # time derivative
                Mform_u = u/dT * v * dx
                Mform_un = -un/dT * v * dx
                self.Forms.add(Form(Mform_u, sp, "Mu"))
                self.Forms.add(Form(Mform_un, sp, "Mun"))

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
        comp_list = [self.CD.Dict[key] for key in self.u.keys()]
        self.split_forms = ddict(dict)
        form_types = set([f.form_type for f in self.Forms.form_list])

        if self.config.solver['nonlinear'] == 'picard':
            Print("Splitting problem into bilinear and linear forms for picard iterations: a(u,v) == L(v)")
            for comp in comp_list:
                comp_forms = [f.dolfin_form for f in self.Forms.select_by('compartment_name', comp.name)]
                self.a[comp.name] = d.lhs(sum(comp_forms))
                self.L[comp.name] = d.rhs(sum(comp_forms))
                problem = d.LinearVariationalProblem(self.a[comp.name],
                                                     self.L[comp.name], self.u[comp.name]['u'], [])
                self.linear_solver[comp.name] = d.LinearVariationalSolver(problem)
                p = self.linear_solver[comp.name].parameters
                p['linear_solver'] = self.config.solver['linear_solver']
                p['krylov_solver'].update(self.config.dolfin_krylov_solver)
                p['krylov_solver'].update({'nonzero_initial_guess': True}) # important for time dependent problems

        elif self.config.solver['nonlinear'] == 'newton':
            Print("Formulating problem as F(u;v) == 0 for newton iterations")
            for comp in comp_list:
                comp_forms = [f.dolfin_form for f in self.Forms.select_by('compartment_name', comp.name)]
                self.F[comp.name] = sum(comp_forms)
                J = d.derivative(self.F[comp.name], self.u[comp.name]['u'])
                problem = d.NonlinearVariationalProblem(self.F[comp.name], self.u[comp.name]['u'], [], J)

                self.nonlinear_solver[comp.name] = d.NonlinearVariationalSolver(problem)
                p = self.nonlinear_solver[comp.name].parameters
                p['nonlinear_solver'] = 'newton'

                p['newton_solver'].update(self.config.dolfin_nonlinear_solver)
                p['newton_solver']['krylov_solver'].update(self.config.dolfin_krylov_solver)
                p['newton_solver']['krylov_solver'].update({'nonzero_initial_guess': True}) # important for time dependent problems


        elif self.config.solver['nonlinear'] == 'IMEX':
            raise Exception("IMEX functionality needs to be reviewed")
#            Print("Keeping forms separated by compartment and form_type for IMEX scheme.")
#            for comp in comp_list:
#                comp_forms = self.Forms.select_by('compartment_name', comp.name)
#                for form_type in form_types:
#                    self.split_forms[comp.name][form_type] = sum([f.dolfin_form for f in comp_forms if f.form_type==form_type])

#===============================================================================
#===============================================================================
# SOLVING
#===============================================================================
#===============================================================================

    def solve(self, op_split_scheme="DRD", plot_period=1):
        ## solve
        self.init_solver_and_plots()

        self.stopwatch("Total simulation")
        while True:
            # Solve using specified operator-splitting scheme (just DRD for now)
            if op_split_scheme == "DRD":
                self.DRD_solve(boundary_method='RK45')
            elif op_split_scheme == "DR":
                self.DR_solve(boundary_method='RK45')
            else:
                raise Exception("I don't know what operator splitting scheme to use")

            self.compute_statistics()
            if self.idx % plot_period == 0 or self.t >= self.config.solver['T']:
                self.plot_solution()
                self.plot_solver_status()
            if self.t >= self.config.solver['T']:
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

        # if last time-step we passed a reset dt checkpoint then reset it now
        if self.reset_dt:
            new_dt = self.config.advanced['reset_dt'][0]
            color_print("(!!!) Adjusting time-step (dt = %f -> %f) to match config specified value" % (self.dt, new_dt), 'green')
            self.set_time(self.t, dt = new_dt)
            self.config.advanced['reset_dt'] = self.config.advanced['reset_dt'][1:]
            self.reset_dt = False
            return

        # check that there are reset times specified
        if hasattr(self.config, 'advanced'):
            if 'reset_times' in self.config.advanced.keys():
                if len(self.config.advanced['reset_times'])!=len(self.config.advanced['reset_dt']):
                    raise Exception("The number of times to reset dt must be equivalent to the length of the list of dts to reset to.")
                if len(self.config.advanced['reset_times']) == 0:
                    return
            else:
                return
        else:
            return

        # check if we pass a reset dt checkpoint
        t0 = self.t # original time
        potential_t = t0 + self.dt # the final time if dt is not reset
        next_reset_time = self.config.advanced['reset_times'][0] # the next time value to reset dt at
        next_reset_dt = self.config.advanced['reset_dt'][0] # next value of dt to reset to
        if next_reset_time<t0:
            raise Exception("Next reset time is less than time at beginning of time-step.")
        if t0 < next_reset_time <= potential_t: # check if the full time-step would pass a reset dt checkpoint
            new_dt = max([next_reset_time - t0, next_reset_dt]) # this is needed otherwise very small time-steps might be taken which wont converge
            color_print("(!!!) Adjusting time-step (dt = %f -> %f) to avoid passing reset dt checkpoint" % (self.dt, new_dt), 'blue')
            self.set_time(self.t, dt=new_dt)
            self.config.advanced['reset_times'] = self.config.advanced['reset_times'][1:]
            # set a flag to change dt to the config specified value
            self.reset_dt = True


    def forward_time_step(self, factor=1):

        self.dT.assign(float(self.dt*factor))
        self.t = float(self.t+self.dt*factor)
        self.T.assign(self.t)
        #print("t: %f , dt: %f" % (self.t, self.dt*factor))

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

    def updateTimeDependentParameters(self, t=None, t0=None, dt=None):
        if not t:
            t = self.t
        if t0 and dt:
            raise Exception("Specify either t0 or dt, not both.")
        elif t0 != None:
            dt = t-t0
        elif dt != None:
            t0 = t-dt
        if t0 != None:
            if t0<0 or dt<0:
                raise Exception("Either t0 or dt is less than 0, is this the desired behavior?")

        # Update time dependent parameters
        for param_name, param in self.PD.Dict.items():
            # check to make sure a parameter wasn't assigned a new value more than once
            value_assigned = 0
            if not param.is_time_dependent:
                continue
            # use value by evaluating symbolic expression
            if param.symExpr and not param.preintegrated_symExpr:
                newValue = param.symExpr.subs({'t': t}).evalf()
                value_assigned += 1

            # calculate a preintegrated expression by subtracting previous value
            # and dividing by time-step
            if param.symExpr and param.preintegrated_symExpr:
                if t0 == None:
                    raise Exception("Must provide a time interval for"\
                                    "pre-integrated variables.")
                newValue = (param.preintegrated_symExpr.subs({'t': t}).evalf()
                            - param.preintegrated_symExpr.subs({'t': t0}).evalf())/dt
                value_assigned += 1

            # if parameter is given by data
            if param.sampling_data is not None and param.preint_sampling_data is None:
                data = param.sampling_data
                # We never want time to extrapolate beyond the provided data.
                if t<data[0,0] or t>data[-1,0]:
                    raise Exception("Parameter cannot be extrapolated beyond"\
                                    "provided data.")
                # Just in case... return a nan if value is outside of bounds
                newValue = np.interp(t, data[:,0], data[:,1],
                                     left=np.nan, right=np.nan)
                value_assigned += 1

            # if parameter is given by data and it has been pre-integrated
            if param.sampling_data is not None and param.preint_sampling_data is not None:
                int_data = param.preint_sampling_data
                oldValue = np.interp(t0, int_data[:,0], int_data[:,1],
                                     left=np.nan, right=np.nan)
                newValue = (np.interp(t, int_data[:,0], int_data[:,1],
                                     left=np.nan, right=np.nan) - oldValue)/dt
                value_assigned += 1

            if value_assigned != 1:
                raise Exception("Either a value was not assigned or more than"\
                                "one value was assigned to parameter %s" % param.name)

            param.value = newValue
            param.dolfinConstant.assign(newValue)
            Print('%f assigned to time-dependent parameter %s'
                  % (newValue, param.name))
            self.params[param_name].append((t,newValue))


    def reset_timestep(self, comp_list=[]):
        """
        Resets the time back to what it was before the time-step. Optionally, input a list of compartments
        to have their function values reset (['n'] value will be assigned to ['u'] function).
        """
        self.set_time(self.t - self.dt, self.dt*self.config.solver['dt_decrease_factor'])
        Print("Resetting time-step and decreasing step size")
        for comp_name in comp_list:
            self.u[comp_name]['u'].assign(self.u[comp_name]['n'])
            Print("Assigning old value of u to species in compartment %s" % comp_name)

    def update_solution_boundary_to_volume(self, keys=['k', 'u']):
        for sp_name, sp in self.SD.Dict.items():
            if sp.parent_species:
                sp_parent = self.SD.Dict[sp.parent_species]
                submesh_species_index = sp.compartment_index
                idx = sp_parent.dof_map[sp_name]

                pcomp_name = sp_parent.compartment_name
                comp_name = sp.compartment_name

                for key in keys:

                    self.u[pcomp_name][key].vector()[idx] = \
                        stubs.data_manipulation.dolfinGetFunctionValues(self.u[comp_name]['u'], self.V[comp_name], submesh_species_index)

                Print("Assigned values from %s (%s) to %s (%s) [keys: %s]" % (sp_name, comp_name, sp_parent.name, pcomp_name, keys))

    def update_solution_volume_to_boundary(self):
        for comp_name in self.CD.Dict.keys():
            for key in self.u[comp_name].keys():
                if key[0] == 'b':
                    self.u[comp_name][key].interpolate(self.u[comp_name]['u'])
                    sub_comp_name = key[1:]
                    Print("Projected values from volume %s to surface %s" % (comp_name, sub_comp_name))


    def boundary_reactions_forward(self, factor=1, bcs=[], key='n'):
        self.stopwatch("Boundary reactions forward")
        nsubsteps = 1#int(self.config.solver['reaction_substeps'])

        for n in range(nsubsteps):
            self.forward_time_step(factor=factor/nsubsteps)
            self.updateTimeDependentParameters(t0=self.t)
            for comp_name, comp in self.CD.Dict.items():
                if comp.dimensionality < self.CD.max_dim:
                    self.nonlinear_solve(comp_name, factor=factor)
                    self.set_time(self.t-self.dt)
                    self.u[comp_name][key].assign(self.u[comp_name]['u'])
        self.stopwatch("Boundary reactions forward", stop=True)

    # def diffusion_forward(self, comp_name, factor=1, bcs=[]):
    #     self.stopwatch("Diffusion step ["+comp_name+"]")
    #     t0 = self.t
    #     self.forward_time_step(factor=factor)
    #     self.updateTimeDependentParameters(t0=t0)
    #     if self.config.solver['nonlinear'] == 'picard':
    #         self.picard_loop(comp_name, bcs)
    #     elif self.config.solver['nonlinear'] == 'newton':
    #         self.newton_iter(comp_name)
    #     self.u[comp_name]['n'].assign(self.u[comp_name]['u'])
    #     self.stopwatch("Diffusion step ["+comp_name+"]", stop=True)


    def nonlinear_solve(self, comp_name, factor=1.0):
        """
        A switch for choosing a nonlinear solver
        """
        if self.config.solver['nonlinear'] == 'newton':
            self.NLidx, success = self.newton_iter(comp_name, factor=factor)
        elif self.config.solver['nonlinear'] == 'picard':
            self.NLidx, success = self.picard_loop(comp_name, factor=factor)

        return self.NLidx, success


    def DRD_solve(self, bcs=[], boundary_method='RK45'):
        """
        General DRD operator splitting. Can be used with different non-linear
        solvers
        """
        Print('\n\n\n')
        self.idx += 1
        self.check_dt_resets()
        color_print('\n *** Beginning time-step %d [time=%f, dt=%f] ***\n' % (self.idx, self.t, self.dt), color='red')

        self.stopwatch("Total time step")

        # solve volume problem(s)
        for comp_name, comp in self.CD.Dict.items():
            if comp.dimensionality == self.CD.max_dim:
                #self.NLidx, success = self.newton_iter(comp_name, factor=0.5)
                self.NLidx, success = self.nonlinear_solve(comp_name, factor=0.5)
                self.set_time(self.t-self.dt/2) # reset time back to t=t0
        self.update_solution_volume_to_boundary()

        # single iteration
        # solve boundary problem(s)
        self.boundary_reactions_forward(factor=1) # t from [t0, t+dt]. automatically resets time back to t0
        self.update_solution_boundary_to_volume()
        self.set_time(self.t-self.dt/2) # perform the second half-step

        # solve volume problem(s)
        for comp_name, comp in self.CD.Dict.items():
            if comp.dimensionality == self.CD.max_dim:
                self.NLidx, success = self.nonlinear_solve(comp_name, factor=0.5)
                self.set_time(self.t-self.dt/2)
        self.set_time(self.t+self.dt/2)
        self.update_solution_volume_to_boundary()

        # check if time step should be changed
        if self.NLidx <= self.config.solver['min_newton']:
            self.set_time(self.t, dt=self.dt*self.config.solver['dt_increase_factor'])
            Print("Increasing step size")
        if self.NLidx > self.config.solver['max_newton']:
            self.set_time(self.t, dt=self.dt*self.config.solver['dt_decrease_factor'])
            Print("Decreasing step size")

        self.stopwatch("Total time step", stop=True, color='cyan')

    def DR_solve(self, bcs=[], boundary_method='RK45'):
        """
        General DR operator splitting. Can be used with different non-linear
        solvers
        """
        Print('\n\n\n')
        self.idx += 1
        self.check_dt_resets()
        color_print('\n *** Beginning time-step %d [time=%f, dt=%f] ***\n' % (self.idx, self.t, self.dt), color='red')

        self.stopwatch("Total time step")

        # solve volume problem(s)
        for comp_name, comp in self.CD.Dict.items():
            if comp.dimensionality == self.CD.max_dim:
                self.NLidx, success = self.nonlinear_solve(comp_name, factor=1.0)
                self.set_time(self.t-self.dt) # reset time back to t=t0
        self.update_solution_volume_to_boundary()

        # single iteration
        # solve boundary problem(s)
        self.boundary_reactions_forward(factor=1) # t from [t0, t+dt]. automatically resets time back to t0
        self.update_solution_boundary_to_volume()

        # check if time step should be changed
        if self.NLidx <= self.config.solver['min_newton']:
            self.set_time(self.t, dt=self.dt*self.config.solver['dt_increase_factor'])
            Print("Increasing step size")
        if self.NLidx > self.config.solver['max_newton']:
            self.set_time(self.t, dt=self.dt*self.config.solver['dt_decrease_factor'])
            Print("Decreasing step size")

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

    def newton_iter(self, comp_name, factor=1, bcs=[], assign_to_n=True):
        """
        A single iteration of Newton's method for a single component.
        """
        self.stopwatch("Newton's method [%s]" % comp_name)
        t0 = self.t
        self.forward_time_step(factor=factor) # increment time afterwards
        self.updateTimeDependentParameters(t0=t0)

        idx, success = self.nonlinear_solver[comp_name].solve()

        if assign_to_n:
            self.u[comp_name]['n'].assign(self.u[comp_name]['u'])

        self.stopwatch("Newton's method [%s]" % comp_name, stop=True)
        Print("%d Newton iterations required for convergence." % idx)
        return idx, success

    def picard_loop(self, comp_name, factor=1, bcs=[], assign_to_n=True):
        """
        Continue picard iterations until a specified tolerance or count is
        reached.
        """
        self.stopwatch("Picard loop [%s]" % comp_name)
        t0 = self.t
        self.forward_time_step(factor=factor) # increment time afterwards
        self.updateTimeDependentParameters(t0=t0)
        self.pidx = 0 # count the number of picard iterations
        success = True

        # main loop
        while True:
            self.pidx += 1
            #linear_solver_settings = self.config.dolfin_krylov_solver

            # solve
            self.linear_solver[comp_name].solve()
            #d.solve(self.a[comp_name]==self.L[comp_name], self.u[comp_name]['u'], bcs, solver_parameters=linear_solver_settings)

            # update temporary value of u
            self.data.computeError(self.u, comp_name, self.config.solver['norm'])
            self.u[comp_name]['k'].assign(self.u[comp_name]['u'])

            # Exit if error tolerance or max iterations is reached
            Print('Linf norm (%s) : %f ' % (comp_name, self.data.errors[comp_name]['Linf']['abs'][-1]))
            if self.data.errors[comp_name]['Linf']['abs'][-1] < self.config.solver['picard_abstol']:
                #print("Norm (%f) is less than linear_abstol (%f), exiting picard loop." %
                #(self.data.errors[comp_name]['Linf'][-1], self.config.solver['linear_abstol']))
                break
#            if self.data.errors[comp_name]['Linf']['rel'][-1] < self.config.solver['linear_reltol']:
#                print("Norm (%f) is less than linear_reltol (%f), exiting picard loop." %
#                (self.data.errors[comp_name]['Linf']['rel'][-1], self.config.solver['linear_reltol']))
#                break

            if self.pidx >= self.config.solver['max_picard']:
                Print("Max number of picard iterations reached (%s), exiting picard loop with abs error %f." %
                (comp_name, self.data.errors[comp_name]['Linf']['abs'][-1]))
                success = False
                break

        self.stopwatch("Picard loop [%s]" % comp_name, stop=True)

        if assign_to_n:
            self.u[comp_name]['n'].assign(self.u[comp_name]['u'])

        Print("%d Picard iterations required for convergence." % self.pidx)
        return self.pidx, success


#===============================================================================
#===============================================================================
# POST-PROCESSING
#===============================================================================
#===============================================================================
    def init_solver_and_plots(self):
        self.data.initSolutionFiles(self.SD, write_type='xdmf')
        self.data.storeSolutionFiles(self.u, self.t, write_type='xdmf')
        self.data.computeStatistics(self.u, self.t, self.dt, self.SD, self.PD, self.CD, self.FD, self.NLidx)
        self.data.initPlot(self.config, self.SD, self.FD)

    def update_solution(self):
        for key in self.u.keys():
            self.u[key]['n'].assign(self.u[key]['u'])
            self.u[key]['k'].assign(self.u[key]['u'])

    def compute_statistics(self):
        self.data.computeStatistics(self.u, self.t, self.dt, self.SD, self.PD, self.CD, self.FD, self.NLidx)
        self.data.computeProbeValues(self.u, self.t, self.dt, self.SD, self.PD, self.CD, self.FD, self.NLidx)
        self.data.outputPickle(self.config)
        self.data.outputCSV(self.config)

    def plot_solution(self):
        self.data.storeSolutionFiles(self.u, self.t, write_type='xdmf')
        self.data.plotParameters(self.config)
        self.data.plotSolutions(self.config, self.SD)
        self.data.plotFluxes(self.config)

    def plot_solver_status(self):
        self.data.plotSolverStatus(self.config)
