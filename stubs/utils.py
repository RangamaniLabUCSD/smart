
# ====================================================
# fancy printing
# ====================================================

import os
from pandas import read_json


__all__ = ["json_to_ObjectContainer"]


# # demonstrate built in options
# def _fancy_print_options():
#     for format_type in ['title', 'subtitle', 'log', 'log_important', 'log_urgent', 'timestep', 'solverstep']:
#         _fancy_print(format_type, format_type=format_type)


# ====================================================
# I/O
# ====================================================


def json_to_ObjectContainer(json_str, data_type=None):
    """
    Converts a json_str (either a string of the json itself, or a filepath to
    the json)
    """
    if not data_type:
        raise Exception(
            "Please include the type of data this is (parameters, species, compartments, reactions)."
        )

    if json_str[-5:] == ".json":
        if not os.path.exists(json_str):
            raise Exception("Cannot find JSON file, %s" % json_str)

    df = read_json(json_str).sort_index()
    df = nan_to_none(df)
    if data_type in ["parameters", "parameter", "param", "p"]:
        return ParameterContainer(df)
    elif data_type in ["species", "sp", "spec", "s"]:
        return SpeciesContainer(df)
    elif data_type in ["compartments", "compartment", "comp", "c"]:
        return CompartmentContainer(df)
    elif data_type in ["reactions", "reaction", "r", "rxn"]:
        return ReactionContainer(df)
    else:
        raise Exception(
            "I don't know what kind of ObjectContainer this .json file should be"
        )
