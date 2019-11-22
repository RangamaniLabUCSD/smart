import os
import sys
sys.path.append("../../")
import stubs
from collections import defaultdict
from copy import deepcopy

class ParameterSweep:
    def __init__(self):
        self.names = set()
        self.params = {'': stubs.model_building.ParameterDF()}

    # Same as ParameterDF.append(), but values is a list of possible values.
    def append(self, name, values, unit, group, notes='', is_time_dependent=False,
               sampling_file='', sampling_data=None, dolfinConstant=None,
               symExpr=None, preint_sampling_data=None, preintegrated_symExpr=None):
        # Remove duplicates.
        values = list(set(values))

        # Check if we already added this name.
        if name in self.names:
            raise Exception(f'Name {name} already added to ParameterSweep.')
        self.names.add(name)

        # Take the old params, and create a new ParameterDF for each new value.
        cur_params = self.params
        self.params = dict()
        for new_value in values:
            for fp, cur_param in cur_params.items():
                new_param = deepcopy(cur_param)
                new_param.append(name,
                                 new_value,
                                 unit,
                                 group,
                                 notes,
                                 is_time_dependent,
                                 sampling_file,
                                 sampling_data,
                                 dolfinConstant,
                                 symExpr,
                                 preint_sampling_data,
                                 preintegrated_symExpr)
                self.params[f'{fp}/{name}={new_value}'] = new_param

    def write_json(self, cwd='', name='parameters.json'):
        for fp, param in self.params.items():
            try:
                os.makedirs(f'{cwd}{fp}')
            except OSError:
                print (f'Creation of the directory {cwd}{fp} failed')
            else:
                fp = f'{cwd}{fp}{name}'
                param.df.to_json(fp)
                print(f'Parameters generated successfully! Saved as {fp}')
