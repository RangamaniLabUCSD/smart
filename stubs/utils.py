# ====================================================
# fancy printing
# ====================================================
import os

import pandas
from pandas import read_json

from .model_assembly import CompartmentContainer
from .model_assembly import ParameterContainer
from .model_assembly import ReactionContainer
from .model_assembly import SpeciesContainer

__all__ = ["json_to_ObjectContainer", "write_sbmodel", "read_sbmodel", "empty_sbmodel"]


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
            "Please include the type of data this is (parameters, species, compartments, reactions).",
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
            "I don't know what kind of ObjectContainer this .json file should be",
        )


# def write_sbmodel(filepath, pdf, sdf, cdf, rdf):
#     """
#     Takes a ParameterDF, SpeciesDF, CompartmentDF, and ReactionDF, and generates
#     a .sbmodel file (a convenient concatenation of .json files with syntax
#     similar to .xml)
#     """
#     f = open(filepath, "w")

#     f.write("<sbmodel>\n")
#     # parameters
#     f.write("<parameters>\n")
#     pdf.df.to_json(f)
#     f.write("\n</parameters>\n")
#     # species
#     f.write("<species>\n")
#     sdf.df.to_json(f)
#     f.write("\n</species>\n")
#     # compartments
#     f.write("<compartments>\n")
#     cdf.df.to_json(f)
#     f.write("\n</compartments>\n")
#     # reactions
#     f.write("<reactions>\n")
#     rdf.df.to_json(f)
#     f.write("\n</reactions>\n")

#     f.write("</sbmodel>\n")
#     f.close()
#     print(f"sbmodel file saved successfully as {filepath}!")


def write_sbmodel(filepath, pc, sc, cc, rc):
    """
    Takes a ParameterDF, SpeciesDF, CompartmentDF, and ReactionDF, and generates
    a .sbmodel file (a convenient concatenation of .json files with syntax
    similar to .xml)
    """
    f = open(filepath, "w")

    f.write("<sbmodel>\n")
    # parameters
    f.write("<parameters>\n")
    pdf.df.to_json(f)
    f.write("\n</parameters>\n")
    # species
    f.write("<species>\n")
    sdf.df.to_json(f)
    f.write("\n</species>\n")
    # compartments
    f.write("<compartments>\n")
    cdf.df.to_json(f)
    f.write("\n</compartments>\n")
    # reactions
    f.write("<reactions>\n")
    rdf.df.to_json(f)
    f.write("\n</reactions>\n")

    f.write("</sbmodel>\n")
    f.close()
    print(f"sbmodel file saved successfully as {filepath}!")


def read_sbmodel(filepath, output_type=dict):
    f = open(filepath)
    lines = f.read().splitlines()
    if lines[0] != "<sbmodel>":
        raise Exception(f"Is {filepath} a valid .sbmodel file?")

    p_string = []
    c_string = []
    s_string = []
    r_string = []
    line_idx = 0

    while True:
        if line_idx >= len(lines):
            break
        line = lines[line_idx]
        if line == "</sbmodel>":
            print("Finished reading in sbmodel file")
            break

        if line == "<parameters>":
            print("Reading in parameters")
            while True:
                line_idx += 1
                if lines[line_idx] == "</parameters>":
                    break
                p_string.append(lines[line_idx])

        if line == "<species>":
            print("Reading in species")
            while True:
                line_idx += 1
                if lines[line_idx] == "</species>":
                    break
                s_string.append(lines[line_idx])

        if line == "<compartments>":
            print("Reading in compartments")
            while True:
                line_idx += 1
                if lines[line_idx] == "</compartments>":
                    break
                c_string.append(lines[line_idx])

        if line == "<reactions>":
            print("Reading in reactions")
            while True:
                line_idx += 1
                if lines[line_idx] == "</reactions>":
                    break
                r_string.append(lines[line_idx])

        line_idx += 1

    pdf = pandas.read_json("".join(p_string)).sort_index()
    sdf = pandas.read_json("".join(s_string)).sort_index()
    cdf = pandas.read_json("".join(c_string)).sort_index()
    rdf = pandas.read_json("".join(r_string)).sort_index()
    pc = stubs.model_assembly.ParameterContainer(nan_to_none(pdf))
    sc = stubs.model_assembly.SpeciesContainer(nan_to_none(sdf))
    cc = stubs.model_assembly.CompartmentContainer(nan_to_none(cdf))
    rc = stubs.model_assembly.ReactionContainer(nan_to_none(rdf))

    if output_type == dict:
        return {
            "parameter_container": pc,
            "species_container": sc,
            "compartment_container": cc,
            "reaction_container": rc,
        }
    elif output_type == tuple:
        return (pc, sc, cc, rc)


# def create_sbmodel(p, s, c, r, output_type=dict):
#     pc = stubs.model_assembly.ParameterContainer(p)
#     sc = stubs.model_assembly.SpeciesContainer(s)
#     cc = stubs.model_assembly.CompartmentContainer(c)
#     rc = stubs.model_assembly.ReactionContainer(r)

#     if output_type==dict:
#         return {'parameter_container': pc,   'species_container': sc,
#                 'compartment_container': cc, 'reaction_container': rc}
#     elif output_type==tuple:
#         return (pc, sc, cc, rc)


def empty_sbmodel():
    pc = ParameterContainer()
    sc = SpeciesContainer()
    cc = CompartmentContainer()
    rc = ReactionContainer()
    return pc, sc, cc, rc
