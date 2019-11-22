import sys
sys.path.append("../../")
import stubs
from copy import deepcopy

# Takes the same arguments as ParameterDF.append(), except that values contains
# a dictionary of names to their respective value lists. Returns a dictionary
# mapping top-level relative file paths to their ParameterDF dataframes.
def param_sweep(values, unit, group, notes='', is_time_dependent=False,
                sampling_file='', sampling_data=None, dolfinConstant=None,
                symExpr=None, preint_sampling_data=None, preintegrated_symExpr=None):
    # Remove any duplicates.
    for name in values:
        values[name] = list(set(values[name]))

    # Use backtracking to get all combinations of parameters.
    params = [[]]
    for name in values:
        cur_params = params
        params = list()
        for value in values[name]:
            for cur_param in cur_params:
                # Add a tuple of name, value for each combination.
                cur_param.append((name, value))
                params.append(deepcopy(cur_param))
                cur_param.pop()

    # Create a dataframe and a filepath from a params entry.
    def param_list_to_dict(pl):
        df = stubs.model_building.ParameterDF()
        fp = ''
        for name, value in pl:
            fp += f'{name}={value}/'
            df.append(name,
                      value,
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
        fp += 'parameters.json'
        return fp, df

    param_dict = {}
    for param in params:
        fp, df = param_list_to_dict(param)
        param_dict[fp] = df

    return param_dict