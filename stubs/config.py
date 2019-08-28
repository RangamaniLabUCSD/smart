# parse the .config file regex parser
import re


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
        print("im done reading")



