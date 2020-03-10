import re
import os
from pandas import read_json
import dolfin as d
from stubs.common import nan_to_none
from stubs.common import round_to_n
from stubs import model_assembly
from stubs import unit as ureg
import random
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import petsc4py.PETSc as PETSc
Print = PETSc.Sys.Print

import mpi4py.MPI as pyMPI
comm = d.MPI.comm_world
rank = comm.rank
size = comm.size
root = 0

class Config(object):
    """
    A class which loads in data from a configuration file, and a model from
    parameter/species/compartment/reaction jsons. The config settings are
    applied and the model object is generated via generate_model()
    """
    def __init__(self, config_file=None):
        self._regex_dict = {
            'comment': re.compile(r'\#.*\n'),
            #'setting_string': re.compile(r'\$\s*(?P<group>\w*).(?P<parameter>\w*)\s*=\s*(?P<value>[A-z|\/]\S*)'),
            'setting_list': re.compile(r'\$\s(?P<group>\w*).(?P<parameter>\w*)\s*=\s*\[(?P<value>.*)\]'),
            'setting_string': re.compile(r'\$\s*(?P<group>\w*).(?P<parameter>\w*)\s*=\s*(?P<value>[A-Za-z_|()\/]\S*)'),
            'setting_float': re.compile(r'\$\s*(?P<group>\w*).(?P<parameter>\w*)\s*=\s*(?P<value>[\d.e+-]\S*)'),
            #'float': re.compile(r'[\s\'\,]*(?P<value>[\d.e+-]*)[\s\'\,]*'),
            'float': re.compile(r'\b(?P<value>[\d.e+-]+)\b'),
            'string': re.compile(r'\b(?P<value>[A-Za-z_]+[\dA-Za-z_]*)\b'),
            'xml_vertex': re.compile(r'.*vertex index=.*x=\"(?P<value_x>[\d.e+-]*)\" y=\"(?P<value_y>[\d.e+-]*)\" z=\"(?P<value_z>[\d.e+-]*)\".*')
            }

        if not config_file:
            Print("Warning: no configuration file specified.")
        else:
            self.config_file = config_file
            self._parse_file()

            # prepend a parent directory to file paths
            if 'parent' in self.directory.keys():
                if self.directory['relative'] == True:
                    dirname = os.path.relpath(self.directory['parent'])
                else:
                    dirname = os.path.abspath("/"+self.directory['parent'])
                if rank==root and not os.path.exists(dirname):
                        os.mkdir(dirname)
                for key, item in self.directory.items():
                    if key not in ['parent', 'relative']:
                        self.directory[key] = dirname + '/' + item 

            self.settings['ignore_surface_diffusion'] = True if self.settings['ignore_surface_diffusion'] == 'True' else False
            self.settings['add_boundary_species'] = True if self.settings['add_boundary_species'] == 'True' else False
            self.settings['zero_d'] = True if self.settings['zero_d'] == 'True' else False


    def _parse_list(self,a_list):
        a_list = list(a_list.split())
        for idx, item in enumerate(a_list):
            key, match = self._parse_line(item)
            if key == 'float':
                a_list[idx] = float(match.group('value'))
            elif key =='string':
                a_list[idx] = str(match.group('value'))
        return a_list

    def _parse_line(self,line):
        for key, regex in self._regex_dict.items():
            match = regex.search(line)
            if match:
                return key, match
        return None, None

    def _parse_file(self):
        with open(self.config_file, 'r') as file:
            line = file.readline()
            while line:
                key, match = self._parse_line(line)

                if key in ['setting_string', 'setting_float', 'setting_list']:
                    group = match.group('group')
                    parameter = match.group('parameter')
                    value = match.group('value')
                    # initialize an empty dict
                    if not hasattr(self,group):
                        setattr(self, group, {})
                else:
                    line = file.readline()
                    continue
                
                if key == 'setting_string':
                    new_value = value
                if key == 'setting_float':
                    new_value = float(value)
                if key == 'setting_list':
                    new_value = self._parse_list(value)

                # most parameters will be caught by the regex but some we may wish to redefine
                # change to int
                if parameter in ['maximum_iterations']:
                    Print("Defining parameter %s as an int\n" % parameter)
                    new_value = int(value)
                # change to bool
                if parameter in ['error_on_nonconvergence', 'nonzero_initial_guess', 'relative']:
                    Print("Defining parameter %s as a bool\n" % parameter)
                    new_value = bool(float(value))


                if key in ['setting_string', 'setting_float', 'setting_list']:
                    getattr(self,group)[parameter] = new_value

                line = file.readline()


            if 'directory' in self.model.keys():
                model_dir = self.model['directory']
                Print("\nAssuming file names, loading from directory %s" % model_dir)
                self.model['parameters'] = model_dir + 'parameters.json'
                self.model['compartments'] = model_dir + 'compartments.json'
                self.model['species'] = model_dir + 'species.json'
                self.model['reactions'] = model_dir + 'reactions.json'

        if (all([x in self.model.keys() for x in ['parameters', 'species', 'compartments', 'reactions']]) 
            and self.mesh.keys()):
            Print("Parameters, species, compartments, reactions, and a mesh were imported succesfully!")

        file.close()

    def find_mesh_midpoint(self,filename):
        m = d.Mesh(filename)
        return (m.coordinates().max(axis=0) - m.coordinates().min(axis=0))/2

    def find_mesh_COM(self,filename):
        m = d.Mesh(filename)
        return m.coordinates().mean(axis=0)

        # FIXME
    # def transform_coords_xml(self, scaling_factor, full_filename, sig_figs=10,
    #                       new_midpoint=None, max_num_vert_midpoint=1e9,
    #                       pre_rotation=(0,0,0), post_rotation=(0,0,0)):
    #     """
    #     Transform the coordinates a dolfin xml file by some scaling factor with
    #     options to rotate and translate the mesh as well, e.g. to center the
    #     center of mass at the origin.
    #     $$t' = $$
    #     $$x_new = R_{post}(sR_{pre}(x) + t')\bar{x}$$
        
    #     Args:
    #         scaling_factor (TYPE): scale length by this factor
    #         full_filename (TYPE): xml dolfin mesh
    #         sig_figs (int, optional): Number of sig figs for coordinates
    #         new_midpoint (3-tuple [float], optional): If the entire mesh
    #         should be translated a tuple can be provided which will be the new
    #         midpoint of the mesh.
    #         max_num_vert_midpoint (int, optional): Maximum number of vertices 
    #         to sample when computing midpoint
    #         pre_rotation (3-tuple [float], optional): Rotation to apply to mesh
    #         before translation/scaling [xyz, given in degrees]
    #         post_rotation (3-tuple [float], optional): Rotation to apply to mesh
    #         after translation/scaling [xyz, given in degrees]
    
    #     """
    #     file_dir, file_name = os.path.split(os.path.abspath(full_filename))
    #     new_full_filename = (file_dir + '/' + file_name.split('.')[0] + '_scaled.' 
    #                         + '.'.join(file_name.split('.')[1:]))
    #     new_file_lines = [] # we will append modified lines to here and write out as a new file
    #     idx = 0 
    #     Rpre = Rot.from_euler('xyz', pre_rotation,degrees=True)
    #     Rpost = Rot.from_euler('xyz', post_rotation,degrees=True)

    #     if new_midpoint is not None:
    #         midpoint = self.find_mesh_midpoint(full_filename, max_num_vert=max_num_vert_midpoint)
    #         # translates from the original midpoint to new midpoint
    #         translation_vector = np.array(new_midpoint) - scaling_factor*Rpre.apply(midpoint)
    #         Print((f"Mesh translated by {translation_vector} to new midpoint, "
    #               + f"{new_midpoint}"))
    #     else:
    #         translation_vector = np.array([0,0,0]) # no translation
    #     with open(full_filename, 'r') as file:
    #         line = file.readline()
    #         while line:
    #             match = self._regex_dict['xml_vertex'].search(line)

    #             if match:
    #                 # parse the original vector
    #                 old_vector = np.ndarray(3)
    #                 for coord_idx, coord in enumerate(['x','y','z']):
    #                     old_vector[coord_idx] = float(match.group('value_'+coord))
    #                     # new_value = (float(match.group('value_'+coord))
    #                     #              + translation_vector[coord_idx]) * scaling_factor
    #                 # apply transformations (pre-rotation, translate+scale,
    #                 # post-rotation)
    #                 #new_vector = Rpre.apply(old_vector)
    #                 #new_vector = (Rpre.apply(old_vector))*scaling_factor + translation_vector
    #                 new_vector = scaling_factor*Rpre.apply(old_vector - midpoint) + np.array(new_midpoint)
    #                 #new_vector = Rpost.apply(new_vector)
    #                 # write the transformed vector to file
    #                 for coord_idx, coord in enumerate(['x','y','z']):
    #                     new_value = round_to_n(new_vector[coord_idx], sig_figs)
    #                     line = re.sub(' '+coord+r'=\"[\d.e+-]+\"', ' '+coord+'=\"'+str(new_value)+'\"', line)

    #             new_file_lines.append(line)
    #             line = file.readline()
    #             idx += 1
    #             if idx%10000==0: # print every 10000 so it is readable
    #                 Print('Finished parsing line %d' % idx)

    #     # once we're done modifying we write out to a new file
    #     with open(new_full_filename, 'w+') as file:
    #         file.writelines(new_file_lines)
    #     Print("Scaled mesh is saved as %s" % new_full_filename)


    def generate_model(self):
        if (self.settings['zero_d']):
            return self.generate_ode_model()

        if not all([x in self.model.keys() for x in ['parameters', 'species', 'compartments', 'reactions']]):
            raise Exception("Parameters, species, compartments, and reactions must all be specified.")
        PD = self._json_to_ObjectContainer(self.model['parameters'], 'parameters')
        SD = self._json_to_ObjectContainer(self.model['species'], 'species')
        CD = self._json_to_ObjectContainer(self.model['compartments'], 'compartments')
        RD = self._json_to_ObjectContainer(self.model['reactions'], 'reactions')

        # parameter/unit assembly
        PD.do_to_all('assemble_units', {'unit_name': 'unit'})
        PD.do_to_all('assemble_units', {'value_name':'value', 'unit_name':'unit', 'assembled_name': 'value_unit'})
        PD.do_to_all('assembleTimeDependentParameters')
        SD.do_to_all('assemble_units', {'unit_name': 'concentration_units'})
        SD.do_to_all('assemble_units', {'unit_name': 'D_units'})
        CD.do_to_all('assemble_units', {'unit_name':'compartment_units'})
        RD.do_to_all('initialize_flux_equations_for_known_reactions', {"reaction_database": self.reaction_database})


        # linking containers with one another
        RD.link_object(PD,'paramDict','name','paramDictValues', value_is_key=True)
        SD.link_object(CD,'compartment_name','name','compartment')
        SD.copy_linked_property('compartment', 'dimensionality', 'dimensionality')
        RD.do_to_all('get_involved_species_and_compartments', {"SD": SD, "CD": CD})
        RD.link_object(SD,'involved_species','name','involved_species_link')
        #RD.do_to_all('combineDicts', {'dict1': 'paramDictValues', 'dict2': 'involved_species_link', 'new_dict_name': 'varDict'})

        # meshes
        CD.add_property('meshes', self.mesh)
        CD.load_mesh('cyto', self.mesh['cyto'])
        CD.extract_submeshes('cyto', False)
        CD.compute_scaling_factors()

        num_species_per_compartment = RD.get_species_compartment_counts(SD, CD, self.settings)
        CD.get_min_max_dim()
        SD.assemble_compartment_indices(RD, CD, self.settings)
        CD.add_property_to_all('is_in_a_reaction', False)
        CD.add_property_to_all('V', None)

        #RD.replace_sub_species_in_reactions(SD)
        #CD.Print()

        # # # dolfin
        SD.assemble_dolfin_functions(RD, CD, self.settings)
        SD.assign_initial_conditions()

        RD.reaction_to_fluxes()
        RD.do_to_all('reaction_to_fluxes')
        FD = RD.get_flux_container()
        FD.do_to_all('get_additional_flux_properties', {"CD": CD, "config": self})

        # # opportunity to make custom changes

        FD.do_to_all('flux_to_dolfin', {"config": self})
        FD.check_and_replace_sub_species(SD, CD, self)

        model = model_assembly.Model(PD, SD, CD, RD, FD, self)

        # to deal with possible floating point error in mesh coordinates
        model.set_allow_extrapolation()
        # Turn fluxes into fenics/dolfin expressions
        model.assemble_reactive_fluxes()
        model.assemble_diffusive_fluxes()
        #model.establish_mappings()
        # Sort forms by type (diffusive, time derivative, etc.)
        model.sort_forms()

        if rank==root:
            Print("Model created succesfully! :)")
            model.PD.print()
            model.SD.print()
            model.CD.print()
            model.RD.print()
            model.FD.print()

        return model

    def generate_ode_model(self):
        if not all([x in self.model.keys() for x in ['parameters', 'species', 'compartments', 'reactions']]):
            raise Exception("Parameters, species, compartments, and reactions must all be specified.")
        PD = self._json_to_ObjectContainer(self.model['parameters'], 'parameters')
        SD = self._json_to_ObjectContainer(self.model['species'], 'species')
        CD = self._json_to_ObjectContainer(self.model['compartments'], 'compartments')
        RD = self._json_to_ObjectContainer(self.model['reactions'], 'reactions')

        CD.compute_scaling_factors()

        # define a base concentration unit
        base_compartment = ""
        base_units = ureg.uM
        for name, species in SD.Dict.items():
            if getattr(species, 'ref'):
                base_compartment = species.compartment_name
                base_units = species.concentration_units
                print(f"Base units defined as {base_units} by reference species {name}.")
                break

        for name, compartment in CD.Dict.items():
            if getattr(compartment, 'dimensionality') != 3:
                setattr(compartment, 'dimensionality', 3)
        
        for name, species in SD.Dict.items():
            if species.compartment_name != base_compartment:
                # CD.Dict[species.compartment_name].print()
                # length_scaling = CD.Dict[species.compartment_name].scale_to(base_compartment)
                length_scaling = CD.Dict[species.compartment_name].scale_to[base_compartment]
                updated_concentration = ureg.Quantity(species.initial_condition, species.concentration_units) * length_scaling
                updated_concentration.ito(base_units)
                setattr(species, 'concentration_units', updated_concentration.units)
                setattr(species, 'initial_condition', updated_concentration.magnitude)

        # parameter/unit assembly
        PD.do_to_all('assemble_units', {'unit_name': 'unit'})
        PD.do_to_all('assemble_units', {'value_name':'value', 'unit_name':'unit', 'assembled_name': 'value_unit'})
        PD.do_to_all('assembleTimeDependentParameters', {'zero_d': True})
        SD.do_to_all('assemble_units', {'unit_name': 'concentration_units'})
        SD.do_to_all('assemble_units', {'unit_name': 'D_units'})
        CD.do_to_all('assemble_units', {'unit_name':'compartment_units'})
        RD.do_to_all('initialize_flux_equations_for_known_reactions', {"reaction_database": self.reaction_database})

        # linking containers with one another
        RD.link_object(PD,'paramDict','name','paramDictValues', value_is_key=True)
        SD.link_object(CD,'compartment_name','name','compartment')
        SD.copy_linked_property('compartment', 'dimensionality', 'dimensionality')
        RD.do_to_all('get_involved_species_and_compartments', {"SD": SD, "CD": CD})
        RD.link_object(SD,'involved_species','name','involved_species_link')

        num_species_per_compartment = RD.get_species_compartment_counts(SD, CD, self.settings)
        CD.get_min_max_dim() # Check dimensions (using min/max)

        SD.assemble_compartment_indices(RD, CD, self.settings) 
        CD.add_property_to_all('is_in_a_reaction', False)
        CD.add_property_to_all('V', None)

        RD.reaction_to_fluxes()
        RD.do_to_all('reaction_to_fluxes')
        FD = RD.get_flux_container()

        # # opportunity to make custom changes

        model = model_assembly.Model(PD, SD, CD, RD, FD, self)
        
        if rank==root:
            Print("Model created succesfully! :)")
            model.PD.print()
            model.SD.print()
            model.CD.print()
            model.RD.print()
            model.FD.print()

        return model


    def _json_to_ObjectContainer(self, json_file_name, data_type=None):
        if not data_type:
            raise Exception("Please include the type of data this is (parameters, species, compartments, reactions).")
        if not os.path.exists(json_file_name):
            raise Exception("Cannot find JSON file, %s"%json_file_name)
        df = read_json(json_file_name).sort_index()
        df = nan_to_none(df)
        if data_type in ['parameters', 'parameter', 'param', 'p']:
            return model_assembly.ParameterContainer(df)
        elif data_type in ['species', 'sp', 'spec', 's']:
            return model_assembly.SpeciesContainer(df)
        elif data_type in ['compartments', 'compartment', 'comp', 'c']:
            return model_assembly.CompartmentContainer(df)
        elif data_type in ['reactions', 'reaction', 'r', 'rxn']:
            return model_assembly.ReactionContainer(df)
        else: 
            raise Exception("I don't know what kind of ObjectContainer this .json file should be")





