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
            'setting': re.compile(r'\$\s*(?P<group>\w*).(?P<parameter>\w*)\s*=\s*(?P<value>\S*)\n')
            }

        self.model = {}
        self.solver = {}
        self.mesh = {}
        self.directory = {}

        self.parse_file()

    def _parse_line(self,line):
        for key, regex in self._regex_dict.items():
            match = regex.search(line)
            if match:
                return key, match
        return None, None

    def parse_file(self):
        with open(self.config_file, 'r') as file:
            line = file.readline()
            while line:
                key, match = self._parse_line(line)

                if key == 'comment':
                    pass
                if key == 'setting':
                    group = match.group('group')
                    parameter = match.group('parameter')
                    value = match.group('value')
                    getattr(self, group)[parameter] = value

                line = file.readline()

        file.close()

    def json_to_ObjectContainer(self, json_file_name, data_type=None):
        if not data_type:
            raise Exception("Please include the type of data this is (parameters, species, compartments, reactions).")
        df = read_json(json_file_name).sort_index()
        if data_type in ['parameters', 'parameter', 'param', 'p']:
            model_assembly.ParameterContainer
        elif data_type in ['species', 'sp', 'spec', 's']:
            model_assembly.SpeciesContainer
        elif data_type in ['compartments', 'compartment', 'comp', 'c']:
            model_assembly.CompartmentContainer
        elif data_type in ['reactions', 'reaction', 'r', 'rxn']:
            model_assembly.ReactionContainer





