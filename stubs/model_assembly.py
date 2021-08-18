"""
Classes for parameters, species, compartments, reactions, fluxes, and forms
Model class contains functions to efficiently solve a system
"""
import pdb
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

import stubs
import stubs.common as common
from stubs import unit as ureg
color_print = common.color_print

comm = d.MPI.comm_world
rank = comm.rank
size = comm.size
root = 0

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
    def assembleTimeDependentParameters(self):
        #TODO
        if not self.is_time_dependent:
            return
        # Parse the given string to create a sympy expression
        if self.symExpr:
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

    def assemble_compartment_indices(self, RD, CD):
        """
        Adds a column to the species dataframe which indicates the index of a species relative to its compartment
        """
        num_species_per_compartment = RD.get_species_compartment_counts(self, CD)
        for compartment, num_species in num_species_per_compartment.items():
            idx = 0
            comp_species = [sp for sp in self.Dict.values() if sp.compartment_name==compartment]
            for sp in comp_species:
                if sp.is_in_a_reaction:
                    sp.compartment_index = idx
                    idx += 1
                else:
                    print('Warning: species %s is not used in any reactions!' % sp.name)


    def assemble_dolfin_functions(self, RD, CD):
        """
        define dof/solution vectors (dolfin trialfunction, testfunction, and function types) based on number of species appearing in reactions
        IMPORTANT: this function will create additional species on boundaries in order to use operator-splitting later on
        e.g.
        A [cyto] + B [pm] <-> C [pm]
        Since the value of A is needed on the pm there will be a species A_b_pm which is just the values of A on the boundary pm
        """

        # functions to run beforehand as we need their results
        num_species_per_compartment = RD.get_species_compartment_counts(self, CD)
        #CD.get_min_max_dim() # refactor
        self.assemble_compartment_indices(RD, CD)
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

        # now we create boundary functions, i.e. interpolations of functions defined on the volume
        # to function spaces of the surrounding mesh
        V['boundary'] = {}
        for compartment_name, num_species in num_species_per_compartment.items():
            compartmentDim = CD.Dict[compartment_name].dimensionality
            if compartmentDim == CD.max_dim: # mesh may have boundaries
                V['boundary'][compartment_name] = {}
                for mesh_name, mesh in CD.meshes.items():
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
            compartmentDim = CD.Dict[compartment_name].dimensionality
            if compartmentDim == CD.min_dim: # mesh may be a boundary with a connected volume
                V['volume'][compartment_name] = {}
                for mesh_name, mesh in CD.meshes.items():
                    if compartment_name != mesh_name and mesh.topology().dim() > compartmentDim:
                        if num_species == 1:
                            volumeV = d.FunctionSpace(mesh, 'P', 1)
                        else:
                            volumeV = d.VectorFunctionSpace(mesh, 'P', 1, dim=num_species)
                        V['volume'][compartment_name].update({mesh_name: volumeV})
                        u[compartment_name]['v_'+mesh_name] = d.Function(volumeV, name="concentration_uv")

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





class Species(_ObjectInstance):
    def __init__(self, name, Dict=None):
        super().__init__(name, Dict)
        self.sub_species = {} # additional compartments this species may live in in addition to its primary one
        self.is_in_a_reaction = False
        self.is_an_added_species = False
        self.dof_map = {}

class CompartmentContainer(_ObjectContainer):
    def __init__(self, df=None, Dict=None):
        super().__init__(Compartment, df, Dict)
        self.properties_to_print = ['name', 'dimensionality', 'num_species', 'num_vertices', 'cell_marker', 'is_in_a_reaction', 'nvolume']
        self.meshes = {}
        self.vertex_mappings = {} # from submesh -> parent indices
#    def load_mesh(self, mesh_key, mesh_str):
#        self.meshes[mesh_key] = d.Mesh(mesh_str)
#
    def extract_submeshes(self, main_mesh_str='main_mesh', save_to_file=False):
        main_mesh  = self.Dict[main_mesh_str]
        surfaceDim = main_mesh.dimensionality - 1

        self.Dict[main_mesh_str].mesh = self.meshes[main_mesh_str]

        vmesh = self.meshes[main_mesh_str]
        bmesh = d.BoundaryMesh(vmesh, "exterior")

        # Very odd behavior - when bmesh.entity_map() is called together with .array() it will return garbage values. We
        # should only call entity_map once to avoid this
        temp_emap_0  = bmesh.entity_map(0)
        bmesh_emap_0 = deepcopy(temp_emap_0.array())
        temp_emap_n  = bmesh.entity_map(surfaceDim)
        bmesh_emap_n = deepcopy(temp_emap_n.array())

        vmf          = d.MeshFunction("size_t", vmesh, surfaceDim, vmesh.domains())
        bmf          = d.MeshFunction("size_t", bmesh, surfaceDim)
        vmf_combined = d.MeshFunction("size_t", vmesh, surfaceDim, vmesh.domains())
        bmf_combined = d.MeshFunction("size_t", bmesh, surfaceDim)

        # iterate through facets of bmesh (transfer markers from volume mesh function to boundary mesh function)
        for idx, facet in enumerate(d.entities(bmesh,surfaceDim)): 
            vmesh_idx = bmesh_emap_n[idx] # get the index of the face on vmesh corresponding to this face on bmesh
            vmesh_boundarynumber = vmf.array()[vmesh_idx] # get the value of the mesh function at this face
            bmf.array()[idx] = vmesh_boundarynumber # set the value of the boundary mesh function to be the same value

        # combine markers for subdomains specified as a list of markers
        for comp_name, comp in self.Dict.items():
            if type(comp.cell_marker) == list:
                if not all([type(x)==int for x in comp.cell_marker]):
                    raise ValueError("Cell markers were given as a list but not all elements were ints.")

                first_index_marker = comp.cell_marker[0] # combine into the first marker of the list 
                comp.first_index_marker = first_index_marker
                Print(f"Combining markers {comp.cell_marker} (for component {comp_name}) into single marker {first_index_marker}.")
                for marker_value in comp.cell_marker:
                    vmf_combined.array()[vmf.array() == marker_value] = first_index_marker
                    bmf_combined.array()[bmf.array() == marker_value] = first_index_marker
            elif type(comp.cell_marker) == int:
                comp.first_index_marker = comp.cell_marker
                vmf_combined.array()[vmf.array() == comp.cell_marker] = comp.cell_marker
                bmf_combined.array()[bmf.array() == comp.cell_marker] = comp.cell_marker
            else:
                raise ValueError("Cell markers must either be provided as an int or list of ints")


        # Loop through compartments: extract submeshes and integration measures
        for comp_name, comp in self.Dict.items():
            # FEniCS doesn't allow parallelization of SubMeshes. We need
            # SubMeshes because one boundary will often have multiple domains of
            # interest with different species (e.g., PM, ER). By exporting the
            # submeshes in serial we can reload them back in in parallel.

            if comp_name!=main_mesh_str and comp.dimensionality==surfaceDim:
                # # TODO: fix this (parallel submesh)
                # if size > 1: # if we are running in parallel
                #     Print("CPU %d: Loading submesh for %s from file" % (rank, comp_name))
                #     submesh = d.Mesh(d.MPI.comm_self, 'submeshes/submesh_' + comp.name + '_' + str(comp.cell_marker) + '.xml')
                #     self.meshes[comp_name] = submesh
                #     comp.mesh = submesh
                # else:
                submesh = d.SubMesh(bmesh, bmf_combined, comp.first_index_marker)
                self.vertex_mappings[comp_name] = submesh.data().array("parent_vertex_indices", 0)
                self.meshes[comp_name] = submesh
                comp.mesh = submesh

                # # TODO: fix this (parallel submesh)
                # if save_to_file:
                #     Print("Saving submeshes %s for use in parallel" % comp_name)
                #     save_str = 'submeshes/submesh_' + comp.name + '_' + str(comp.cell_marker) + '.xml'
                #     d.File(save_str) << submesh

            # integration measures
            if comp.dimensionality==main_mesh.dimensionality:
                comp.ds = d.Measure('ds', domain=comp.mesh, subdomain_data=vmf_combined, metadata={'quadrature_degree': 3})
                comp.ds_uncombined = d.Measure('ds', domain=comp.mesh, subdomain_data=vmf, metadata={'quadrature_degree': 3})
                comp.dP = None
            elif comp.dimensionality<main_mesh.dimensionality:
                comp.dP = d.Measure('dP', domain=comp.mesh)
                comp.ds = None
            else:
                raise Exception("main_mesh is not a maximum dimension compartment")
            comp.dx = d.Measure('dx', domain=comp.mesh, metadata={'quadrature_degree': 3})

        # Get # of vertices
        for key, mesh in self.meshes.items():
            num_vertices = mesh.num_vertices()
            print('CPU %d: My partition of mesh %s has %d vertices' % (rank, key, num_vertices))
            self.Dict[key].num_vertices = num_vertices

        self.bmesh          = bmesh
        self.bmesh_emap_0   = bmesh_emap_0
        self.bmesh_emap_n   = bmesh_emap_n
        self.mesh_functions = {
                                'vmf': vmf,
                                'bmf': bmf,
                                'vmf_combined': vmf_combined,
                                'bmf_combined': bmf_combined,
                              }

        # # TODO: fix this (parallel submesh)
        # # If we were running in serial to generate submeshes, exit here and
        # # restart in parallel
        # if save_to_file and size==1:
        #     Print("If run in serial, submeshes were saved to file. Run again"\
        #           "in parallel.")
        #     exit()

    def extract_submeshes_refactor(self, parent_mesh):
        # Get minimum and maximum dimensions of meshes being computed on.
        volumeDim  = self.max_dim
        surfaceDim = self.min_dim

        # Check that dimensionality of components and mesh is acceptable
        if (volumeDim - surfaceDim) not in [0,1]:
            raise ValueError("(Highest mesh dimension - smallest mesh dimension) must be either 0 or 1.")
        if volumeDim != parent_mesh.dimensionality:
            raise ValueError(f"Parent mesh has geometric dimension: {parent_mesh.dolfin_mesh.geometric_dimension()} which"
                            +f" is not the same as volumeDim: {volumeDim}.")

        # Get volume and boundary mesh
        #self.Dict[main_mesh_str].mesh   = self.meshes[main_mesh_str]
        smesh           = d.BoundaryMesh(parent_mesh.dolfin_mesh, "exterior")

        # When smesh.entity_map() is called together with .array() it will return garbage values. We should only call 
        # entity_map once to avoid this
        temp_emap_0     = smesh.entity_map(0)
        smesh_emap_0    = deepcopy(temp_emap_0.array()) # entity map to vertices
        temp_emap_n     = smesh.entity_map(surfaceDim)
        smesh_emap_n    = deepcopy(temp_emap_n.array()) # entity map to facets

        # Mesh functions
        # 
        # vvmf: cell markers for volume mesh. Used to distinguish sub-volumes
        #
        #
        #
        #vvmf         = d.MeshFunction("size_t", vmesh, volumeDim, vmesh.domains())  # cell markers for volume mesh
        #vsmf         = d.MeshFunction("size_t", vmesh, surfaceDim, vmesh.domains()) # facet markers for volume mesh
        #smf          = d.MeshFunction("size_t", bmesh, surfaceDim)                  # cell markers for surface mesh
        #vmf_combined = d.MeshFunction("size_t", vmesh, surfaceDim, vmesh.domains()) # 
        #bmf_combined = d.MeshFunction("size_t", bmesh, surfaceDim)

        vmf          = d.MeshFunction("size_t", vmesh, surfaceDim, vmesh.domains())
        bmf          = d.MeshFunction("size_t", bmesh, surfaceDim)
        vmf_combined = d.MeshFunction("size_t", vmesh, surfaceDim, vmesh.domains())
        bmf_combined = d.MeshFunction("size_t", bmesh, surfaceDim)

        # iterate through facets of bmesh (transfer markers from volume mesh function to boundary mesh function)
        for idx, facet in enumerate(d.entities(bmesh,surfaceDim)): 
            vmesh_idx = bmesh_emap_n[idx] # get the index of the face on vmesh corresponding to this face on bmesh
            vmesh_boundarynumber = vmf.array()[vmesh_idx] # get the value of the mesh function at this face
            bmf.array()[idx] = vmesh_boundarynumber # set the value of the boundary mesh function to be the same value

        # combine markers for subdomains specified as a list of markers
        for comp_name, comp in self.Dict.items():
            if type(comp.cell_marker) == list:
                if not all([type(x)==int for x in comp.cell_marker]):
                    raise ValueError("Cell markers were given as a list but not all elements were ints.")

                first_index_marker = comp.cell_marker[0] # combine into the first marker of the list 
                comp.first_index_marker = first_index_marker
                Print(f"Combining markers {comp.cell_marker} (for component {comp_name}) into single marker {first_index_marker}.")
                for marker_value in comp.cell_marker:
                    vmf_combined.array()[vmf.array() == marker_value] = first_index_marker
                    bmf_combined.array()[bmf.array() == marker_value] = first_index_marker
            elif type(comp.cell_marker) == int:
                comp.first_index_marker = comp.cell_marker
                vmf_combined.array()[vmf.array() == comp.cell_marker] = comp.cell_marker
                bmf_combined.array()[bmf.array() == comp.cell_marker] = comp.cell_marker
            else:
                raise ValueError("Cell markers must either be provided as an int or list of ints")


        # Loop through compartments: extract submeshes and integration measures
        for comp_name, comp in self.Dict.items():
            # FEniCS doesn't allow parallelization of SubMeshes. We need
            # SubMeshes because one boundary will often have multiple domains of
            # interest with different species (e.g., PM, ER). By exporting the
            # submeshes in serial we can reload them back in in parallel.

            if comp_name!=main_mesh_str and comp.dimensionality==surfaceDim:
                # # TODO: fix this (parallel submesh)
                # if size > 1: # if we are running in parallel
                #     Print("CPU %d: Loading submesh for %s from file" % (rank, comp_name))
                #     submesh = d.Mesh(d.MPI.comm_self, 'submeshes/submesh_' + comp.name + '_' + str(comp.cell_marker) + '.xml')
                #     self.meshes[comp_name] = submesh
                #     comp.mesh = submesh
                # else:
                submesh = d.SubMesh(bmesh, bmf_combined, comp.first_index_marker)
                self.vertex_mappings[comp_name] = submesh.data().array("parent_vertex_indices", 0)
                self.meshes[comp_name] = submesh
                comp.mesh = submesh

                # # TODO: fix this (parallel submesh)
                # if save_to_file:
                #     Print("Saving submeshes %s for use in parallel" % comp_name)
                #     save_str = 'submeshes/submesh_' + comp.name + '_' + str(comp.cell_marker) + '.xml'
                #     d.File(save_str) << submesh

            # integration measures
            if comp.dimensionality==volumeDim:
                comp.ds = d.Measure('ds', domain=comp.mesh, subdomain_data=vmf_combined, metadata={'quadrature_degree': 3})
                comp.ds_uncombined = d.Measure('ds', domain=comp.mesh, subdomain_data=vmf, metadata={'quadrature_degree': 3})
                comp.dP = None
            elif comp.dimensionality<volumeDim:
                comp.dP = d.Measure('dP', domain=comp.mesh)
                comp.ds = None
            else:
                raise Exception(f"Internal error: {comp_name} has a dimension larger than then the volume dimension.")
            comp.dx = d.Measure('dx', domain=comp.mesh, metadata={'quadrature_degree': 3})

        # Get # of vertices
        for key, mesh in self.meshes.items():
            num_vertices = mesh.num_vertices()
            print('CPU %d: My partition of mesh %s has %d vertices' % (rank, key, num_vertices))
            self.Dict[key].num_vertices = num_vertices

        self.bmesh          = bmesh
        self.bmesh_emap_0   = bmesh_emap_0
        self.bmesh_emap_n   = bmesh_emap_n
        self.mesh_functions = {
                                'vmf': vmf,
                                'bmf': bmf,
                                'vmf_combined': vmf_combined,
                                'bmf_combined': bmf_combined,
                              }

        # # TODO: fix this (parallel submesh)
        # # If we were running in serial to generate submeshes, exit here and
        # # restart in parallel
        # if save_to_file and size==1:
        #     Print("If run in serial, submeshes were saved to file. Run again"\
        #           "in parallel.")
        #     exit()



    def compute_scaling_factors(self):
        self.do_to_all('compute_nvolume')
        for key, comp in self.Dict.items():
            comp.scale_to = {}
            for key2, comp2 in self.Dict.items():
                if key != key2:
                    comp.scale_to.update({key2: comp.nvolume / comp2.nvolume})
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

    def get_species_compartment_counts(self, SD, CD):
        """
        Returns a Counter object with the number of times a species appears in each compartment
        """
        self.do_to_all('get_involved_species_and_compartments', {"SD": SD, "CD": CD})
        all_involved_species = set([sp for species_set in [rxn.involved_species_link.values() for rxn in self.Dict.values()] for sp in species_set])
        for sp_name, sp in SD.Dict.items():
            if sp in all_involved_species:
                sp.is_in_a_reaction = True

        compartment_counts = [sp.compartment_name for sp in all_involved_species]

        return Counter(compartment_counts)

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
                flux_name = self.name + '_f_' + species_name
                sign = -1 if species_name in self.LHS else 1
                signed_stoich = sign*stoich
                self.fluxList.append(Flux(flux_name, species_name, self.eqn_f, signed_stoich, self.involved_species_link,
                                          self.paramDictValues, self.group, self, self.explicit_restriction_to_domain, track))
            if hasattr(self, 'eqn_r'):
                flux_name = self.name + '_r_' + species_name
                sign = 1 if species_name in self.LHS else -1
                signed_stoich = sign*stoich
                self.fluxList.append(Flux(flux_name, species_name, self.eqn_r, signed_stoich, self.involved_species_link,
                                          self.paramDictValues, self.group, self, self.explicit_restriction_to_domain, track))



class FluxContainer(_ObjectContainer):
    def __init__(self, df=None, Dict=None):
        super().__init__(Flux, df, Dict)
        # self.properties_to_print = ['species_name', 'symEqn', 'sign', 'involved_species',
        #                      'involved_parameters', 'source_compartment',
        #                      'destination_compartment', 'ukeys', 'group']

        self.properties_to_print = ['species_name', 'symEqn', 'signed_stoich', 'ukeys']#'source_compartment', 'destination_compartment', 'ukeys']

class Flux(_ObjectInstance):
    def __init__(self, flux_name, species_name, symEqn, signed_stoich,
                 spDict, paramDict, group, parent_reaction=None,
                 explicit_restriction_to_domain=None, track_value=False):
        super().__init__(flux_name)

        self.flux_name = flux_name
        self.species_name = species_name
        self.symEqn = symEqn
        self.signed_stoich = signed_stoich
        self.spDict = spDict
        self.paramDict = paramDict
        self.group = group
        self.parent_reaction = parent_reaction
        self.explicit_restriction_to_domain = explicit_restriction_to_domain
        self.track_value = track_value
        self.tracked_values = []

        self.symList = [str(x) for x in symEqn.free_symbols]
        self.lambdaEqn = sympy.lambdify(self.symList, self.symEqn, modules=['sympy','numpy'])
        self.involved_species = list(spDict.keys())
        self.involved_parameters = list(paramDict.keys())


    def get_additional_flux_properties(self, CD, solver_system):
        # get additional properties of the flux
        self.get_involved_species_parameters_compartment(CD)
        self.get_flux_dimensionality()
        self.get_boundary_marker()
        self.get_flux_units()
        self.get_is_linear()
        self.get_is_linear_comp()
        self.get_ukeys(solver_system)
        self.get_integration_measure(CD, solver_system)

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
            self.boundary_marker = self.involved_compartments[self.source_compartment].first_index_marker

    def get_flux_units(self):
        sp = self.spDict[self.species_name]
        compartment_units = sp.compartment.compartment_units
        # a boundary flux
        if (self.boundary_marker and self.flux_dimensionality[1]>self.flux_dimensionality[0]):
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

    def get_integration_measure(self, CD, solver_system):
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
        flux_vars = [str(x) for x in self.symList if str(x) in self.involved_species]
        for var_name in flux_vars:
            self.ukeys[var_name] = self.get_ukey(var_name, solver_system)

    def get_ukey(self, var_name, solver_system):
        sp = self.spDict[self.species_name]
        var = self.spDict[var_name]

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
                return 'b_'+sp.compartment_name
            # surface -> volume is covered by first if statement in get_ukey()

        raise Exception("Missing logic in get_ukey(); contact a developer...")

    def flux_to_dolfin(self):
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
