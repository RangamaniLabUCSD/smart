# parse the .config file regex parser
import re
import os
from pandas import read_json
import dolfin as d
from stubs.common import nan_to_none
from stubs.common import round_to_n
from stubs import model_assembly
import random
import numpy as np

import mpi4py.MPI as pyMPI

comm = d.MPI.comm_world
rank = comm.rank
size = comm.size
root = 0

class Config(object):
    def __init__(self, config_file=None):
        self._regex_dict = {
            'comment': re.compile(r'\#.*\n'),
            #'setting_string': re.compile(r'\$\s*(?P<group>\w*).(?P<parameter>\w*)\s*=\s*(?P<value>[A-z|\/]\S*)'),
            'setting_list': re.compile(r'\$\s(?P<group>\w*).(?P<parameter>\w*)\s*=\s*\[(?P<value>.*)\]'),
            'setting_string': re.compile(r'\$\s*(?P<group>\w*).(?P<parameter>\w*)\s*=\s*(?P<value>[A-z|()\/]\S*)'),
            'setting_float': re.compile(r'\$\s*(?P<group>\w*).(?P<parameter>\w*)\s*=\s*(?P<value>[\d.e+-]\S*)'),
            'float': re.compile(r'[\s\'\,]*(?P<value>[\d.e+-]*)[\s\'\,]*'),
            'string': re.compile(r'[\s\'\,]*(?P<value>[A-z|()\/]*)[\s\'\,]*'),
            'xml_vertex': re.compile(r'.*vertex index=.*x=\"(?P<value_x>[\d.e+-]*)\" y=\"(?P<value_y>[\d.e+-]*)\" z=\"(?P<value_z>[\d.e+-]*)\".*')
            }

        if not config_file:
            print("No configuration file specified...")
        else:
            self.config_file = config_file
            self._parse_file()
            # we pack this into its own dictionary so it can be input as a parameter into dolfin solve
            self.dolfin_linear = {}
            if 'linear_solver' in self.solver.keys():
                self.dolfin_linear['linear_solver'] = self.solver['linear_solver']
            if 'preconditioner' in self.solver.keys():
                self.dolfin_linear['preconditioner'] = self.solver['preconditioner']
            if 'linear_maxiter' in self.solver.keys():
                self.dolfin_linear['maximum_iterations'] = self.solver['linear_maxiter']

            # prepend a parent directory to file paths
            if 'parent' in self.directory.keys():
                dirname = self.directory['parent']
                if rank==root and not os.path.exists(dirname):
                        os.mkdir(dirname)
                for key, item in self.directory.items():
                    if key != 'parent':
                        self.directory[key] = dirname + '/' + item 

            self.settings['ignore_surface_diffusion'] = True if self.settings['ignore_surface_diffusion'] == 'True' else False
            self.settings['add_boundary_species'] = True if self.settings['add_boundary_species'] == 'True' else False

        self.dolfin_linear = {}
        self.dolfin_linear_coarse = {}


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

                if key == 'comment':
                    pass
                if key in ['setting_string', 'setting_float', 'setting_list']:
                    group = match.group('group')
                    parameter = match.group('parameter')
                    value = match.group('value')
                    # initialize an empty dict
                    if not hasattr(self,group):
                        setattr(self, group, {})

                if key == 'setting_string':
                    new_value = value
                if key == 'setting_float':
                    new_value = float(value)
                if key == 'setting_list':
                    new_value = self._parse_list(value)

                if key in ['setting_string', 'setting_float', 'setting_list']:
                    getattr(self,group)[parameter] = new_value


                line = file.readline()

            if 'directory' in self.model.keys():
                model_dir = self.model['directory']
                if rank==root: print("Assuming file names, loading from directory %s" % model_dir)
                self.model['parameters'] = model_dir + 'parameters.json'
                self.model['compartments'] = model_dir + 'compartments.json'
                self.model['species'] = model_dir + 'species.json'
                self.model['reactions'] = model_dir + 'reactions.json'

        if (all([x in self.model.keys() for x in ['parameters', 'species', 'compartments', 'reactions']]) 
            and self.mesh.keys()):
            print("Parameters, species, compartments, reactions, and a mesh were imported succesfully!")

        file.close()


    def _find_mesh_midpoint(self, filename, max_num_vert=20000, frac=1):
        """
        Finds the (approximate) midpoint of a mesh so that we may center it around the origin
        """
        m = d.Mesh(filename)
        total_num_vert = m.num_vertices()
        sample_num_vert = min([max_num_vert, total_num_vert*frac])
        indices = random.sample(range(total_num_vert), sample_num_vert) # choose a subset of indices
        verts = list(d.vertices(m))

        xmax = verts[0].point().array() # init
        xmin = verts[0].point().array()
        for idx in indices:
            x = verts[idx].point().array()
            xmax = np.maximum(xmax, x)
            xmin = np.minimum(xmin, x)

        midpoint = (xmax-xmin)/2
        print("Found approximate midpoint %s using %d points" % (str(midpoint), sample_num_vert))
        return midpoint


    def _scale_coords_xml(self, scaling_factor, filename, sig_figs=10, move_midpoint_to_origin=False, max_num_vert_midpoint=20000):
        """
        Scales the coordinates a dolfin xml file by some scaling factor
        """
        new_file_lines = [] # we will append modified lines to here and write out as a new file
        new_filename = 'scaled_' + filename 
        idx = 0 
        if move_midpoint_to_origin:
            midpoint = self._find_mesh_midpoint(filename, max_num_vert=max_num_vert_midpoint)
        with open(filename, 'r') as file:
            line = file.readline()
            while line:
                match = self._regex_dict['xml_vertex'].search(line)

                if match:
                    for coord_idx, coord in enumerate(['x','y','z']):
                        if move_midpoint_to_origin:
                            new_value = (float(match.group('value_'+coord)) - midpoint[coord_idx]) * scaling_factor
                        else:
                            new_value = float(match.group('value_'+coord)) * scaling_factor
                        new_value = round_to_n(new_value, sig_figs)

                        line = re.sub(' '+coord+r'=\"[\d.e+-]+\"', ' '+coord+'=\"'+str(new_value)+'\"', line)
                new_file_lines.append(line)
                line = file.readline()
                idx += 1
                if idx%10000==0: # every 10000 so the output is readable
                    print('Finished parsing line %d' % idx)

        # once we're done modifying we write out to a new file
        with open(new_filename, 'w+') as file:
            file.writelines(new_file_lines)
        print("Scaled mesh is saved as %s" % new_filename)


    def generate_model(self):

        if not all([x in self.model.keys() for x in ['parameters', 'species', 'compartments', 'reactions']]):
            raise Exception("Parameters, species, compartments, and reactions must all be specified.")
        PD = self._json_to_ObjectContainer(self.model['parameters'], 'parameters')
        SD = self._json_to_ObjectContainer(self.model['species'], 'species')
        CD = self._json_to_ObjectContainer(self.model['compartments'], 'compartments')
        RD = self._json_to_ObjectContainer(self.model['reactions'], 'reactions')

        # parameter/unit assembly
        PD.doToAll('assemble_units', {'unit_name': 'unit'})
        PD.doToAll('assemble_units', {'value_name':'value', 'unit_name':'unit', 'assembled_name': 'value_unit'})
        PD.doToAll('assembleTimeDependentParameters')
        SD.doToAll('assemble_units', {'unit_name': 'concentration_units'})
        SD.doToAll('assemble_units', {'unit_name': 'D_units'})
        CD.doToAll('assemble_units', {'unit_name':'compartment_units'})
        RD.doToAll('initialize_flux_equations_for_known_reactions', {"reaction_database": self.reaction_database})


        # linking containers with one another
        RD.link_object(PD,'paramDict','name','paramDictValues', value_is_key=True)
        SD.link_object(CD,'compartment_name','name','compartment')
        SD.copy_linked_property('compartment', 'dimensionality', 'dimensionality')
        RD.doToAll('get_involved_species_and_compartments', {"SD": SD, "CD": CD})
        RD.link_object(SD,'involved_species','name','involved_species_link')
        #RD.doToAll('combineDicts', {'dict1': 'paramDictValues', 'dict2': 'involved_species_link', 'new_dict_name': 'varDict'})

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
        #CD.print()

        # # # dolfin
        SD.assemble_dolfin_functions(RD, CD, self.settings)
        SD.assign_initial_conditions()

        RD.reaction_to_fluxes()
        RD.doToAll('reaction_to_fluxes')
        FD = RD.get_flux_container()
        FD.doToAll('get_additional_flux_properties', {"CD": CD, "config": self})

        # # opportunity to make custom changes


        FD.doToAll('flux_to_dolfin', {"config": self})
        FD.check_and_replace_sub_species(SD, CD, self)

        model = model_assembly.Model(PD, SD, CD, RD, FD, self)

        if rank==root:
            print("Model created succesfully! :)")
            model.PD.print()
            model.SD.print()
            model.CD.print()
            model.RD.print()
            model.FD.print()

        return model


    def _json_to_ObjectContainer(self, json_file_name, data_type=None):
        if not data_type:
            raise Exception("Please include the type of data this is (parameters, species, compartments, reactions).")
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





