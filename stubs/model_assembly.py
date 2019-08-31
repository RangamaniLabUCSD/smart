from collections import Counter
from collections import OrderedDict as odict
from collections import defaultdict as ddict
from termcolor import colored
import pandas as pd
import dolfin as d
import sympy
from sympy.parsing.sympy_parser import parse_expr
import pint
from pprint import pprint
from tabulate import tabulate
from copy import copy

import stubs.common as common
from stubs import unit as ureg
import stubs.data_manipulation as data_manipulation
#import stubs.flux_assembly as flux_assembly



# ====================================================
# ====================================================
# Base Classes
# ====================================================
# ====================================================
class _ObjectContainer(object):
    def __init__(self, ObjectClass, df=None, Dict=None):
        self.Dict = odict()
        self.dtypes = {}
        self.ObjectClass = ObjectClass
        self.propertyList = [] # properties to print
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
#            name = itemDict[self.name_key]
#            itemDict.pop(self.name_key)
#            self.Dict[name] = self.ObjectClass(name, itemDict)
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

        #TODO: instead of linked_name make a dictionary of linked objects
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
                        raise Exception('Either none or more than one objects match this condition')
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
                        raise Exception('Either none or more than one objects match this condition')
                    newDict.update({value: objList[0]})
                setattr(obj1, linked_name, newDict)
            # standard behavior
            else: 
                objList = ObjectContainer2.where_equals(property_name2, obj1_value)
                if len(objList) != 1:
                    raise Exception('Either none or more than one objects match this condition')
                setattr(obj1, linked_name, objList[0])

    def copy_linked_property(self, linked_name, linked_name_property, property_name):
        """
        Convenience function to copy a property from a linked object
        """
        for _, obj in self.Dict.items():
            linked_obj = getattr(obj, linked_name)
            setattr(obj, property_name, getattr(linked_obj, linked_name_property))


    def doToAll(self, method_name, kwargs=None):
        for name, instance in self.Dict.items():
            if kwargs is None:
                getattr(instance, method_name)()
            else:
                getattr(instance, method_name)(**kwargs)

    def get_pandas_dataframe(self, propertyList=[]):
        df = pd.DataFrame()
        if propertyList and 'idx' not in propertyList:
            propertyList.insert(0, 'idx')
        for idx, (name, instance) in enumerate(self.Dict.items()):
            df = df.append(instance.get_pandas_series(propertyList=propertyList, idx=idx))
        # sometimes types are recast. change entries into their original types
        for dtypeName, dtype in self.dtypes.items():
            if dtypeName in df.columns: 
                df = df.astype({dtypeName: dtype})

        return df
    def get_index(self, idx):
        """
        Get an element of the object container ordered dict by referencing its index
        """
        return list(self.Dict.values())[idx]

    def print(self, tablefmt='fancy_grid', propertyList=[]):
        if propertyList:
            if type(propertyList) != list: propertyList=[propertyList]
        elif hasattr(self, 'propertyList'):
            propertyList = self.propertyList
        df = self.get_pandas_dataframe(propertyList=propertyList)
        if propertyList:
            df = df[propertyList]

        print(tabulate(df, headers='keys', tablefmt=tablefmt))#,
               #headers='keys', tablefmt=tablefmt), width=120)

    def __str__(self):
        df = self.get_pandas_dataframe(propertyList=self.propertyList)
        df = df[self.propertyList]

        return tabulate(df, headers='keys', tablefmt='fancy_grid')
        #TODO: look this up
               #headers='keys', tablefmt=tablefmt), width=120)

    def vprint(self, keyList=None, propertyList=[], print_all=False):
        # in order of priority: kwarg, container object property, else print all keys
        if keyList:
            if type(keyList) != list: keyList=[keyList]
        elif hasattr(self, 'keyList'):
            keyList = self.keyList
        else:
            keyList = list(self.Dict.keys())

        if propertyList:
            if type(propertyList) != list: propertyList=[propertyList]
        elif hasattr(self, 'propertyList'):
            propertyList = self.propertyList

        if print_all: propertyList = []
        for key in keyList:
            self.Dict[key].print(propertyList=propertyList)


class _ObjectInstance(object):
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

    def get_pandas_series(self, propertyList=[], idx=None):
        if propertyList:
            dict_to_convert = odict({'idx': idx})
            dict_to_convert.update(odict([(key,val) for (key,val) in self.__dict__.items() if key in propertyList]))
        else:
            dict_to_convert = self.__dict__
        return pd.Series(dict_to_convert, name=self.name)
    def print(self, propertyList=[]):
        print("Name: " + self.name)
        # if a custom list of properties to print is provided, only use those
        if propertyList:
            dict_to_print = dict([(key,val) for (key,val) in self.__dict__.items() if key in propertyList])
        else:
            dict_to_print = self.__dict__
        pprint(dict_to_print, width=240)


# ==============================================================================
# ==============================================================================
# Classes for parameters, species, compartments, reactions, and fluxes
# ==============================================================================
# ==============================================================================

# parameters, compartments, species, reactions, fluxes
class ParameterContainer(_ObjectContainer):
    def __init__(self, df=None, Dict=None):
        super().__init__(Parameter, df, Dict)
        self.propertyList = ['name', 'value', 'unit', 'is_time_dependent', 'symExpr', 'notes', 'group']

class Parameter(_ObjectInstance):
    def __init__(self, name, Dict=None):
        super().__init__(name, Dict)
    def assembleTimeDependentParameters(self): 
        if self.is_time_dependent:
            self.symExpr = parse_expr(self.symExpr)
            print("Creating dolfin object for time-dependent parameter %s" % self.name)
            self.dolfinConstant = d.Constant(self.value)


class SpeciesContainer(_ObjectContainer):
    def __init__(self, df=None, Dict=None):
        super().__init__(Species, df, Dict)
        self.propertyList = ['name', 'compartment_name', 'compartment_index', 'concentration_units', 'D', 'initial_condition', 'sub_species', 'group']

    def assemble_compartment_indices(self, RD, CD):
        """
        Adds a column to the species dataframe which indicates the index of a species relative to its compartment
        """
        num_species_per_compartment = RD.get_species_compartment_counts(self, CD)
        for compartment, num_species in num_species_per_compartment.items():
            idx = 0
            comp_species = [sp for sp in self.Dict.values() if sp.compartment_name==compartment]
            for sp in comp_species:
                if sp.is_in_a_reaction or sp.parent_species:
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
        CD.get_min_max_dim()
        self.assemble_compartment_indices(RD, CD)
        CD.add_property_to_all('is_in_a_reaction', False)
        CD.add_property_to_all('V', None)

        V, u, v = {}, {}, {}
        for compartment_name, num_species in num_species_per_compartment.items():
            compartmentDim = CD.Dict[compartment_name].dimensionality
            CD.Dict[compartment_name].num_species = num_species
            print('Compartment %s (dimension: %d) has %d species associated with it' %
                  (compartment_name, compartmentDim, num_species))
        
            # u is the actual function. t is for linearized versions. k is for picard iterations. n is for last time-step solution
            if num_species == 1:
                V[compartment_name] = d.FunctionSpace(CD.meshes[compartment_name], 'P', 1)
                u[compartment_name] = {'u': d.Function(V[compartment_name]), 't': d.TrialFunction(V[compartment_name]),
                'k': d.Function(V[compartment_name]), 'n': d.Function(V[compartment_name])}
                v[compartment_name] = d.TestFunction(V[compartment_name])
            else: # vector space
                V[compartment_name] = d.VectorFunctionSpace(CD.meshes[compartment_name], 'P', 1, dim=num_species)
                u[compartment_name] = {'u': d.Function(V[compartment_name]), 't': d.TrialFunctions(V[compartment_name]),
                'k': d.Function(V[compartment_name]), 'n': d.Function(V[compartment_name])}
                v[compartment_name] = d.TestFunctions(V[compartment_name])

        # # now we create boundary functions, which are defined on the function spaces of the surrounding mesh
        # V['boundary'] = {}
        # for compartment_name, num_species in num_species_per_compartment.items():
        #     compartmentDim = CD.Dict[compartment_name].dimensionality
        #     if compartmentDim == CD.max_dim: # mesh may have boundaries
        #         for boundary_name, boundary_mesh in CD.meshes.items():
        #             if compartment_name != boundary_name:
        #                 if num_species == 1:
        #                     boundaryV = d.FunctionSpace(CD.meshes[boundary_name], 'P', 1)
        #                 else:
        #                     boundaryV = d.VectorFunctionSpace(CD.meshes[boundary_name], 'P', 1, dim=num_species)
        #                 V['boundary'].update({compartment_name: {boundary_name: boundaryV}})
        #                 u[compartment_name].update({'b': d.Function(boundaryV)})

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

        # associate function spaces with dataframe
        for key, comp in CD.Dict.items():
            if comp.is_in_a_reaction:
                comp.V = V[comp.name]

        self.u = u
        self.v = v
        self.V = V

    def assign_initial_conditions(self):
        keys = ['k', 'n']
        for sp in self.Dict.values():
            comp_name = sp.compartment_name
            for key in keys:
                data_manipulation.dolfinSetFunctionValues(self.u[comp_name][key], sp.initial_condition,
                                                          self.V[comp_name], sp.compartment_index)
            self.u[comp_name]['u'].assign(self.u[comp_name]['n'])
            print("Assigned initial condition to u for species %s" % sp.name)

        # add boundary values
        for comp_name in self.u.keys():
            if 'b' in self.u[comp_name].keys():
                self.u[comp_name]['b'].interpolate(self.u[comp_name]['u'])




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
        self.propertyList = ['name', 'dimensionality', 'num_species', 'num_vertices', 'cell_marker', 'is_in_a_reaction', 'nvolume']
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
        vmf = d.MeshFunction("size_t", vmesh, surfaceDim, vmesh.domains())
        bmf = d.MeshFunction("size_t", bmesh, surfaceDim)
        for idx, facet in enumerate(d.entities(bmesh,surfaceDim)): # iterate through faces of bmesh
            vmesh_idx = bmesh.entity_map(surfaceDim)[idx] # get the index of the face on vmesh corresponding to this face on bmesh
            vmesh_boundarynumber = vmf.array()[vmesh_idx] # get the value of the mesh function at this face
            bmf.array()[idx] = vmesh_boundarynumber # set the value of the boundary mesh function to be the same value


        for key, obj in self.Dict.items():
            if key!=main_mesh_str and obj.dimensionality==surfaceDim:
                submesh = d.SubMesh(bmesh, bmf, obj.cell_marker)                
                self.vertex_mappings[key] = submesh.data().array("parent_vertex_indices", 0)
                self.meshes[key] = submesh
                obj.mesh = submesh
                if save_to_file:
                    save_str = 'submeshes/submesh_' + obj.name + '_' + str(obj.cell_marker) + '.xml'
                    d.File(save_str) << submesh
            # integration measures
            if obj.dimensionality==main_mesh.dimensionality:
                obj.ds = d.Measure('ds', domain=obj.mesh, subdomain_data=vmf)
                obj.dP = None
            elif obj.dimensionality<main_mesh.dimensionality:
                obj.dP = d.Measure('dP', domain=obj.mesh)
                obj.ds = None
            else:
                raise Exception("main_mesh is not a maximum dimension compartment")
            obj.dx = d.Measure('dx', domain=obj.mesh)

        # Get # of vertices
        for key, mesh in self.meshes.items():        
            num_vertices = mesh.num_vertices()
            print('Mesh %s has %d vertices' % (key, num_vertices))
            self.Dict[key].num_vertices = num_vertices

        self.vmf = vmf
        self.bmesh = bmesh
        self.bmf = bmf



    def compute_scaling_factors(self):
        self.doToAll('compute_nvolume')
        for key, obj in self.Dict.items():
            obj.scale_to = {}
            for key2, obj2 in self.Dict.items():
                if key != key2:
                    obj.scale_to.update({key2: obj.nvolume / obj2.nvolume})
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
        self.propertyList = ['name', 'LHS', 'RHS', 'eqn_f', 'eqn_r', 'paramDict', 'reaction_type', 'explicit_restriction_to_domain', 'group']

    def get_species_compartment_counts(self, SD, CD):
        self.doToAll('get_involved_species_and_compartments', {"SD": SD})
        all_involved_species = set([sp for species_set in [rxn.involved_species_link.values() for rxn in self.Dict.values()] for sp in species_set])
        for sp_name, sp in SD.Dict.items():
            if sp in all_involved_species:
                sp.is_in_a_reaction = True

        compartment_counts = [sp.compartment_name for sp in all_involved_species]

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
        for sp_name, comp_name in sub_species_to_add:
            sub_sp_name = sp_name+'_sub_'+comp_name
            compartment_counts.append(comp_name)
            if sub_sp_name not in SD.Dict.keys():
                print((colored('\nSpecies %s will have a new function defined on compartment %s with name: %s\n'
                    % (sp_name, comp_name, sub_sp_name))))

                sub_sp = copy(SD.Dict[sp_name])
                sub_sp.is_an_added_species = True
                sub_sp.name = sub_sp_name
                sub_sp.compartment_name = comp_name
                sub_sp.compartment = CD.Dict[comp_name]
                sub_sp.is_in_a_reaction = True
                sub_sp.sub_species = {}
                sub_sp.parent_species = sp_name
                sub_sp_list.append(sub_sp)

        if sub_sp_name not in SD.Dict.keys():
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
        self.doToAll('reaction_to_fluxes')
        fluxList = []
        for rxn in self.Dict.values():
            for f in rxn.fluxList:
                fluxList.append(f)
        self.fluxList = fluxList
    def get_flux_container(self):
        return FluxContainer(Dict=odict([(f.flux_name, f) for f in self.fluxList]))


class Reaction(_ObjectInstance):
    def __init__(self, name, Dict=None, eqn_f_str=None, eqn_r_str=None):
        if eqn_f_str:
            print(eqn_f_str)
            self.eqn_f = parse_expr(eqn_f_str)
        if eqn_r_str:
            print(eqn_r_str)
            self.eqn_r = parse_expr(eqn_r_str)
        super().__init__(name, Dict)

    def initialize_flux_equations_for_known_reactions(self):
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



    def get_involved_species_and_compartments(self, SD=None):
        # used to get number of active species in each compartment
        self.involved_species = set(self.LHS + self.RHS)
        self.involved_compartments = set([SD.Dict[sp_name].compartment_name for sp_name in self.involved_species])
        if hasattr(self, 'eqn_f'):
            varSet = {str(x) for x in self.eqn_f.free_symbols}
            spSet = varSet.intersection(SD.Dict.keys())
            compSet = set([SD.Dict[sp_name].compartment_name for sp_name in spSet])
            for sp in spSet: self.involved_species.add(sp)
            for comp in compSet: self.involved_compartments.add(comp)
        if hasattr(self, 'eqn_r'):
            varSet = {str(x) for x in self.eqn_r.free_symbols}
            spSet = varSet.intersection(SD.Dict.keys())
            compSet = set([SD.Dict[sp_name].compartment_name for sp_name in spSet])
            for sp in spSet: self.involved_species.add(sp)
            for comp in compSet: self.involved_compartments.add(comp)

        # determine if species is in additional compartments beyond its primary compartment


    def reaction_to_fluxes(self):
        self.fluxList = []
        for species_name in self.LHS + self.RHS:
            if hasattr(self, 'eqn_f'):
                flux_name = self.name + ' (f) [' + species_name + ']'
                sign = -1 if species_name in self.LHS else 1
                self.fluxList.append(Flux(flux_name, species_name, self.eqn_f, sign, self.involved_species_link,
                                          self.paramDictValues, self.group, self.explicit_restriction_to_domain))
            if hasattr(self, 'eqn_r'):
                flux_name = self.name + ' (r) [' + species_name + ']'
                sign = 1 if species_name in self.LHS else -1
                self.fluxList.append(Flux(flux_name, species_name, self.eqn_r, sign, self.involved_species_link,
                                          self.paramDictValues, self.group, self.explicit_restriction_to_domain))





class FluxContainer(_ObjectContainer):
    def __init__(self, df=None, Dict=None):
        super().__init__(Flux, df, Dict)
        self.propertyList = ['species_name', 'symEqn', 'sign', 'involved_species',
                             'involved_parameters', 'source_compartment', 
                             'destination_compartment', 'ukeys', 'group']

    def check_and_replace_sub_species(self, CD, config):
        # TODO: this is a rough fix... find a more robust way to implement this
        # unfortunately this probably has to be done at the flux rather than
        # reaction level...
        fluxes_to_remove = []
        new_flux_list = []
        for flux_name, f in self.Dict.items():
            has_a_sub_species = False
            is_source = False
            sp_to_remove = []
            sp_to_add = []
            for sp_name, sp in f.spDict.items():
                if sp.sub_species and (f.destination_compartment in sp.sub_species.keys()
                                     or f.source_compartment in sp.sub_species.keys()):
                    if f.destination_compartment in sp.sub_species.keys():
                        # flux is volume -> surface
                        sub_sp = sp.sub_species[f.destination_compartment]
                        sp_to_remove.append(sp_name)
                    elif f.source_compartment in sp.sub_species.keys():
                        # flux is surface -> volume
                        sub_sp = sp.sub_species[f.source_compartment]
                        # in this case, we want to create another flux for the sub species
                        is_source = True

                    sub_sp_name = sub_sp.name

                    f.symEqn = f.symEqn.subs({sp_name: sub_sp_name})
                    sp_to_add.append((sub_sp_name, sub_sp))

                    has_a_sub_species = True

            for (sp_add, sp_obj) in sp_to_add:
                f.spDict.update({sp_add: sp_obj})
                print("Modifying flux %s to use subspecies %s!" % (f.flux_name, sp_add))#, sub_sp_name))
            for sp_remove in sp_to_remove:
                f.spDict.pop(sp_remove)

            if has_a_sub_species:
                #new_flux_name = flux_name + ' [sub]'
                #new_f = Flux(f.flux_name + ' [sub]', f.species_name, f.symEqn, f.sign, f.spDict,
                #                          f.paramDict, f.group, f.explicit_restriction_to_domain)
                #new_f.get_additional_flux_properties(CD, config)

                #fluxes_to_remove.append(flux_name)
                #new_flux_list.append((new_flux_name, new_f))

                # in this case, we want to create another flux for the sub species
                if is_source:
                    f.spDict.pop(f.species_name)
                    new_flux_name = f.flux_name + ' [sub_]'
                    new_flux = Flux(f.flux_name + ' [sub_]', sub_sp_name, f.symEqn, f.sign, f.spDict, f.paramDict,
                                    f.group, f.explicit_restriction_to_domain)
                    new_flux.get_additional_flux_properties(CD, config)

                    new_flux_list.append((new_flux_name, new_flux))
        for flux_rm in fluxes_to_remove:
            print('removing flux %s' %  flux_rm)
            self.Dict.pop(flux_rm)

        for (new_flux_name, new_f) in new_flux_list:
            print('adding flux %s' % new_flux_name)
            self.Dict.update({new_flux_name: new_f})



class Flux(_ObjectInstance):
    def __init__(self, flux_name, species_name, symEqn, sign, spDict, paramDict, group, explicit_restriction_to_domain=None):
        super().__init__(flux_name)

        self.flux_name = flux_name
        self.species_name = species_name
        self.symEqn = symEqn
        self.sign = sign
        self.spDict = spDict
        self.paramDict = paramDict
        self.group = group
        self.explicit_restriction_to_domain = explicit_restriction_to_domain

        self.symList = [str(x) for x in symEqn.free_symbols]
        self.lambdaEqn = sympy.lambdify(self.symList, self.symEqn)
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
        self.flux_to_dolfin()

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
        if var.dimensionality < sp.dimensionality: 
            return 'u' # always true if operator splitting to decouple compartments

        if sp.name == var.parent_species:
            print('Debug 1')
            return 'u'
        
        if config.solver['nonlinear'] == 'picard':
            # volume -> surface
            if var.dimensionality > sp.dimensionality:
                if self.is_linear_wrt_comp[var.compartment_name]:
                    return 'bt'
                else:
                    return 'bk'

            # volumetric fluxes
            if var.compartment_name == self.destination_compartment:
                if self.is_linear_wrt_comp[var.compartment_name]:
                    return 't'
                else:
                    return 'k'

        elif config.solver['nonlinear'] == 'newton':
            return 'u'

        raise Exception("If you made it to this far in get_ukey() I missed some logic...") 


    def flux_to_dolfin(self):
        value_dict = {}
        unit_dict = {}

        for var_name in [str(x) for x in self.symList]:
            if var_name in self.paramDict.keys():
                var = self.paramDict[var_name]
                if var.is_time_dependent:
                    value_dict[var_name] = var.dolfinConstant.get()
                else:
                    value_dict[var_name] = var.value_unit.magnitude
                unit_dict[var_name] = var.value_unit.units * 1 # turns unit into "Quantity" class
            elif var_name in self.spDict.keys():
                var = self.spDict[var_name]
                ukey = self.ukeys[var_name]
                if ukey[0] == 'b':
                    if not var.parent_species:
                        sub_species = var.sub_species[self.destination_compartment]
                        value_dict[var_name] = sub_species.u[ukey[1]]
                        print("Species %s substituted for %s in flux %s" % (var_name, sub_species.name, self.name))
                    else:
                        value_dict[var_name] = var.u[ukey[1]]
                else:
                    value_dict[var_name] = var.u[ukey]

                unit_dict[var_name] = var.concentration_units * 1

        prod = self.lambdaEqn(**value_dict)
        unit_prod = self.lambdaEqn(**unit_dict)
        unit_prod = 1 * (1*unit_prod).units # trick to make object a "Quantity" class

        self.prod = prod
        self.unit_prod = unit_prod






# ==============================================================================
# ==============================================================================
# Model class consists of parameters, species, etc. and is used for simulation
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

        self.u = SD.u
        self.v = SD.v
        self.V = SD.V

        self.idx = 0
        self.t = 0.0
        self.dt = config.solver['initial_dt']
        self.T = d.Constant(self.t)
        self.dT = d.Constant(self.dt)
        self.t_final = config.solver['T']

        self.timers = {}
        self.timings = ddict(list)

        self.Forms = FormContainer()
        self.a = {}
        self.L = {}

        self.data = data_manipulation.Data(config)


    def assemble_reactive_fluxes(self):
        """
        Creates the actual dolfin objects for each flux. Checks units for consistency
        """
        for j in self.FD.Dict.values():
            sp = j.spDict[j.species_name]
            prod = j.prod
            unit_prod = j.unit_prod
            # first, check unit consistency
            if (unit_prod/j.flux_units).dimensionless:
                setattr(j, 'scale_factor', 1*ureg.dimensionless)
                pass
            else:
                print(unit_prod)
                print(j.flux_units)
                print(j.name)
                print(j.species_name)
                print('Adjusting flux between compartments %s and %s by length scale factor to ensure consistency' \
                      % tuple(j.involved_compartments.keys()))
                length_scale_factor = j.involved_compartments[j.source_compartment].scale_to[j.destination_compartment]
                if (length_scale_factor*unit_prod/j.flux_units).dimensionless:
                    print('Debug marker 1')
                    prod *= length_scale_factor.magnitude
                    unit_prod *= length_scale_factor.units*1
                    setattr(j, 'length_scale_factor', length_scale_factor)
                elif (1/length_scale_factor*unit_prod/j.flux_units).dimensionless:
                    print('Debug marker 2')
                    prod /= length_scale_factor.magnitude
                    unit_prod /= length_scale_factor.units*1
                    setattr(j, 'length_scale_factor', 1/length_scale_factor)
                else:
                    raise Exception("Inconsitent units!")

            # if units are consistent in dimensionality but not magnitude, adjust values
            if j.flux_units != unit_prod:
                print(('\nThe flux, %s, has units '%j.flux_name + colored(unit_prod, "red") +
                    "...the desired units for this flux are " + colored(j.flux_units, "cyan")))
                unit_scaling = unit_prod.to(j.flux_units).magnitude
                prod *= unit_scaling
                print('Adjusted value of flux by ' + colored("%f"%unit_scaling, "cyan") + ' to match units.\n')
                setattr(j, 'unit_scaling', unit_scaling)
            else:
                setattr(j, 'unit_scaling', 1)

            # adjust sign if necessary
            prod *= j.sign

            # multiply by appropriate integration measure and test function
            # if this is a boundary flux from lower dimension -> higher dimension, multiply by diffusion coefficient
            # (fenics is slightly picky about how these multiplications are carried out)
            if j.flux_dimensionality[0] < j.flux_dimensionality[1]:
                prod = sp.D*prod*sp.v*j.int_measure
                form_key = 'B'
            else:
                prod = prod*sp.v*j.int_measure
                form_key = 'R'

            setattr(j, 'dolfin_flux', prod)

            BRform = -prod
            self.Forms.add(Form(BRform, sp, form_key, flux_name=j.name))


    def assemble_diffusive_fluxes(self):
        min_dim = min(self.CD.get_property('dimensionality').values())
        max_dim = max(self.CD.get_property('dimensionality').values())
        dT = self.dT

        for sp_name, sp in self.SD.Dict.items():
            if sp.is_in_a_reaction:
                u = sp.u['t']
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

                #if sp.dimensionality == max_dim or not self.config.settings['ignore_surface_diffusion']:
                #    dx = sp.compartment.dx
                #    Dform = D*d.inner(d.grad(u), d.grad(v)) * dx
                #    self.Forms.add(Form(Dform, sp, 'D'))
                #elif sp.dimensionality < max_dim and self.config.settings['ignore_surface_diffusion']:
                #    dx = sp.compartment.dP

                # time derivative
                Mform = (u-un)/dT * v * dx
                self.Forms.add(Form(Mform, sp, 'M'))

            else:
                print("Species %s is not in a reaction?" %  sp_name)

    def set_allow_extrapolation(self):
        for comp_name in self.u.keys():
            ucomp = self.u[comp_name] 
            for func_key in ucomp.keys():
                if func_key != 't': # trial function by convention
                    self.u[comp_name][func_key].set_allow_extrapolation(True)

#===============================================================================
#===============================================================================
# SOLVING
#===============================================================================
#===============================================================================
    def set_time(self, t, dt=None):
        if not dt:
            self.dt = dt
            self.dT.assign(dt) 
        self.t = t
        self.T.assign(t)

        print("New time: %f" % self.t)
        print("New dt: %f" % self.dt)

    def forward_time_step(self, factor=1):
        self.dT.assign(float(self.dt*factor))
        self.t = float(self.t+self.dt*factor)
        self.T.assign(self.t)

        print("t: %f , dt: %f" % (self.t, self.dt*factor))

    def stopwatch(self, key, stop=False):
        if key not in self.timers.keys():
            self.timers[key] = d.Timer()
        if not stop:
            self.timers[key].start()
        else:
            elapsed_time = self.timers[key].elapsed()[0]
            self.timers[key].stop()
            self.timings[key].append()
            return elapsed_time

#    def solver_step_forward(self):
#
#        self.update_time()


    def updateTimeDependentParameters(self, t=None): 
        if not t:
            # custom time
            t = self.t
        for param_name, param in self.PD.Dict.items():
            if param.is_time_dependent:
                newValue = param.symExpr.subs({'t': t}).evalf()
                param.dolfinConstant.get().assign(newValue)
                print('%f assigned to time-dependent parameter %s' % (newValue, param.parameter_name))

    def strang_RDR_step_forward(self):
        # first reaction step (half time step) t=[t,t+dt/2]
        for i in range(10):
            self.boundary_reactions_forward()
        print("finished first reaction step")

        # diffusion step (full time step) t=[t,t+dt]
        self.set_time(self.t-self.dt/2, self.dt) # reset time back to t
        self.updateTimeDependentParameters

        # transfer values of solution onto volumetric field
        self.update_solution_boundary_to_volume()

        self.diffusion_forward() 
        #self.SD.Dict['A_sub_pm'].u['u'].interpolate(self.u['cyto']['u'])
        print("finished diffusion step")

        self.update_solution_volume_to_boundary()
        # second reaction step (half time step) t=[t+dt/2,t+dt]
        self.set_time(self.t-self.dt/2, self.dt) # reset time back to t+dt/2
        for i in range(10):
            self.boundary_reactions_forward()
        print("finished second reaction step")


    def establish_mappings(self):
        for sp_name, sp in self.SD.Dict.items():
            if sp.parent_species:
                sp_parent = self.SD.Dict[sp.parent_species]
                Vsub = self.V[sp.compartment_name]
                submesh = self.CD.meshes[sp.compartment_name]
                V = self.V[sp_parent.compartment_name]
                submesh_species_index = sp.compartment_index
                mesh_species_index = sp_parent.compartment_index

                idx = common.submesh_dof_to_mesh_dof(Vsub, submesh, self.CD.bmesh, V,
                                                     submesh_species_index=submesh_species_index,
                                                     mesh_species_index=mesh_species_index)
                sp_parent.dof_map.update({sp_name: idx})

    def update_solution_boundary_to_volume(self):
        for sp_name, sp in self.SD.Dict.items():
            if sp.parent_species:
                sp_parent = self.SD.Dict[sp.parent_species]
                submesh_species_index = sp.compartment_index
                idx = sp_parent.dof_map[sp_name]

                pcomp_name = sp_parent.compartment_name
                comp_name = sp.compartment_name

                self.u[pcomp_name]['u'].vector()[idx] = \
                    data_manipulation.dolfinGetFunctionValues(self.u[comp_name]['u'], self.V[comp_name], submesh_species_index)

                print("Assigned values from %s (%s) to %s (%s)" % (sp_name, comp_name, sp_parent.name, pcomp_name))

    def update_solution_volume_to_boundary(self):
        for sp_name, sp in self.SD.Dict.items():
            if sp.sub_species:
                for comp_name, sp_sub in sp.sub_species.items():
                    sub_name = sp_sub.name
                    submesh_species_index = sp_sub.compartment_index
                    idx = sp.dof_map[sub_name]

                    pcomp_name = sp.compartment_name
                    comp_name = sp_sub.compartment_name

                    unew = self.u[pcomp_name]['u'].vector()[idx]

                    data_manipulation.dolfinSetFunctionValues(self.u[comp_name]['u'], unew, self.V[comp_name], submesh_species_index)
                    print("Assigned values from %s (%s) to %s (%s)" % (sp_name, pcomp_name, sub_name, comp_name))




# for i in range(10):
#     model.boundary_reactions_forward()

# print("finished first reaction step")

# # diffusion step (full time step) t=[t,t+dt]
# model.set_time(model.t-model.dt/2, model.dt) # reset time back to t
# model.updateTimeDependentParameters
# model.diffusion_forward() 
# model.SD.Dict['A_sub_pm'].u['u'].interpolate(model.u['cyto']['u'])
# print("finished diffusion step")

# # second reaction step (half time step) t=[t+dt/2,t+dt]
# self.set_time(self.t-self.dt/2, self.dt) # reset time back to t+dt/2
# for i in range(10):
#     self.boundary_reactions_forward()
# print("finished second reaction step")


    def boundary_reactions_forward(self):
        # first reaction step
        self.forward_time_step(factor=1/20)
        self.updateTimeDependentParameters()
        d.solve(self.a['pm']==self.L['pm'], self.u['pm']['u'])
        self.u['pm']['n'].assign(self.u['pm']['u'])

    def diffusion_forward(self):
        self.forward_time_step()
        self.updateTimeDependentParameters
        d.solve(self.a['cyto']==self.L['cyto'], self.u['cyto']['u'])
        self.u['cyto']['n'].assign(self.u['cyto']['u'])


    def get_lhs_rhs(self):
        # solve the lower dimensional problem first (usually stiffer)
        comp_list = [self.CD.Dict[key] for key in self.u.keys()]
        split_forms = {}

        for comp in comp_list:
            split_forms[comp.name] = [f.dolfin_form for f in self.Forms.select_by('compartment_name', comp.name)]
            self.a[comp.name] = d.lhs(sum(split_forms[comp.name]))
            self.L[comp.name] = d.rhs(sum(split_forms[comp.name]))




#     def init_solver(self):



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
            print("Form with index %d from form_list..." % index)
            if form.flux_name:
                print("Flux name: %s" % form.flux_name)
            print("Species name: %s" % form.species_name)
            print("Form type: %s" % form.form_type)
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
            print(str(integral) + "\n")
