import numpy as np
from pandas import read_json
from .deprecation import deprecated
from .model_assembly import (
    CompartmentContainer,
    ParameterContainer,
    ReactionContainer,
    SpeciesContainer,
)
from typing import Union
from pathlib import Path

__all__ = ["json_to_ObjectContainer"]


@deprecated
def json_to_ObjectContainer(json_file: Union[Path, str], data_type: str):
    """
    Converts a json_str (either a string of the json itself, or a filepath to
    the json) to the appropriate data type (given by a string).

    Args:
        json_file: Path to json file
        data_type: Type of container, either parameter, species, compartment or reaction.

    .. note::
        Several abbreviations of the above are allowed, see source code for details

    """
    json_file = Path(json_file)
    if json_file.suffix != ".json":
        raise ValueError("Invalid suffix for {json_file}, expected '.json'")
    if not json_file.exists():
        raise Exception(f"Cannot find json file: {str(json_file.absolute())}")

    df = read_json(json_file).sort_index()
    df = df.replace({np.nan: None})
    if data_type in ["parameters", "parameter", "param", "p"]:
        return ParameterContainer(df)
    elif data_type in ["species", "sp", "spec", "s"]:
        return SpeciesContainer(df)
    elif data_type in ["compartments", "compartment", "comp", "c"]:
        return CompartmentContainer(df)
    elif data_type in ["reactions", "reaction", "r", "rxn"]:
        return ReactionContainer(df)
    else:
        raise ValueError(f"Unknown data type {data_type} given for {json_file}")
