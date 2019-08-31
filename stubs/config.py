# parse the .config file regex parser
import re
from pandas import read_json
import dolfin as d
from stubs.common import nan_to_none
from stubs import model_assembly



class Config(object):
    def __init__(self, config_file):
        self.config_file = config_file
        self._regex_dict = {
            'comment': re.compile(r'\#.*\n'),
            'setting_string': re.compile(r'\$\s*(?P<group>\w*).(?P<parameter>\w*)\s*=\s*(?P<value>[A-z|\/]\S*)'),
            'setting_float': re.compile(r'\$\s*(?P<group>\w*).(?P<parameter>\w*)\s*=\s*(?P<value>\d\S*)')
            }

        self.model = {}
        self.settings = {}
        self.solver = {}
        self.mesh = {}
        self.directory = {}
        self.plot = {}

        self._parse_file()

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
                if key == 'setting_string':
                    group = match.group('group')
                    parameter = match.group('parameter')
                    value = match.group('value')
                    getattr(self, group)[parameter] = value
                if key == 'setting_float':
                    group = match.group('group')
                    parameter = match.group('parameter')
                    value = match.group('value')
                    getattr(self, group)[parameter] = float(value)

                line = file.readline()

        if (all([x in self.model.keys() for x in ['parameters', 'species', 'compartments', 'reactions']]) 
            and self.mesh.keys()):
            print("Parameters, species, compartments, reactions, and a mesh were imported succesfully!")

        file.close()

    def generate_model(self):

        if not all([x in self.model.keys() for x in ['parameters', 'species', 'compartments', 'reactions']]):
            raise Exception("Parameters, species, compartments, and reactions must all be specified.")
        PD = self._json_to_ObjectContainer(self.model['parameters'], 'parameters')
        SD = self._json_to_ObjectContainer(self.model['species'], 'species')
        CD = self._json_to_ObjectContainer(self.model['compartments'], 'compartments')
        RD = self._json_to_ObjectContainer(self.model['reactions'], 'reactions')

        # parameter/unit assembly
        PD.doToAll('assemble_units', {'value_name':'value', 'unit_name':'unit', 'assembled_name': 'value_unit'})
        PD.doToAll('assembleTimeDependentParameters')
        SD.doToAll('assemble_units', {'unit_name': 'concentration_units'})
        SD.doToAll('assemble_units', {'unit_name': 'D_units'})
        CD.doToAll('assemble_units', {'unit_name':'compartment_units'})
        RD.doToAll('initialize_flux_equations_for_known_reactions')

        # linking containers with one another
        RD.link_object(PD,'paramDict','name','paramDictValues', value_is_key=True)
        SD.link_object(CD,'compartment_name','name','compartment')
        SD.copy_linked_property('compartment', 'dimensionality', 'dimensionality')
        RD.doToAll('get_involved_species_and_compartments', {"SD": SD})
        RD.link_object(SD,'involved_species','name','involved_species_link')
        #RD.doToAll('combineDicts', {'dict1': 'paramDictValues', 'dict2': 'involved_species_link', 'new_dict_name': 'varDict'})

        # meshes
        #CD.add_property('meshes', self.mesh)
        CD.load_mesh('cyto', self.mesh['cyto'])
        CD.extract_submeshes('cyto', True)
        CD.compute_scaling_factors()

        num_species_per_compartment = RD.get_species_compartment_counts(SD, CD)
        CD.get_min_max_dim()
        SD.assemble_compartment_indices(RD, CD)
        CD.add_property_to_all('is_in_a_reaction', False)
        CD.add_property_to_all('V', None)

        #RD.replace_sub_species_in_reactions(SD)


        # dolfin
        SD.assemble_dolfin_functions(RD, CD)
        SD.assign_initial_conditions()

        RD.reaction_to_fluxes()
        RD.doToAll('reaction_to_fluxes')
        FD = RD.get_flux_container()
        FD.doToAll('get_additional_flux_properties', {"CD": CD, "config": self})
        FD.check_and_replace_sub_species(SD, CD, self)

        model = model_assembly.Model(PD, SD, CD, RD, FD, self)

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





