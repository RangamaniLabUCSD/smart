from libsbml import *
from sbml_unit_conversion import *

import sys
import os
import os.path

import stubs
unit = stubs.unit

# taken from https://www.ebi.ac.uk/biomodels/BIOMD0000000932#Files
fp = "models/Garde2020.xml"

# taken from https://www.ebi.ac.uk/biomodels/BIOMD0000000448#Files
# fp = "models/BIOMD0000000448_url.xml"

# taken from https://www.ebi.ac.uk/biomodels/BIOMD0000000931#Files
# fp = "models/Voliotis2019.xml"

doc = readSBML(fp)
if doc.getNumErrors() > 0:
    raise Exception('Unable to read SBML from ' + fp)

c_df = stubs.model_building.CompartmentDF()
p_df = stubs.model_building.ParameterDF()
r_df = stubs.model_building.ReactionDF()
s_df = stubs.model_building.SpeciesDF()

def sbml_to_compartments(document, compartment_df):
    marker = 1
    model = document.getModel()
    compartments = model.getListOfCompartments()
    for compartment in compartments:
        s_id = compartment.getId()
        u_def = compartment.getDerivedUnitDefinition()
        units = sbml_unit_definition_to_pint_quantity(u_def)
        dim = compartment.getSpatialDimensions()
        compartment_df.append(s_id, dim, units.units, marker)
        marker += 1
    return

def sbml_to_parameters(document, parameter_df):
    model = document.getModel()
    parameters = model.getListOfParameters()
    for parameter in parameters:
        s_id = parameter.getId()
        value = float(parameter.getValue())
        u_def = parameter.getDerivedUnitDefinition()
        units = sbml_unit_definition_to_pint_quantity(u_def)
        # Conversion to SI units
        value *= units.magnitude
        notes = parameter.getNotesString()
        parameter_df.append(s_id, value, units.units, notes)
    return

def sbml_to_reactions(document, reaction_df):
    model = document.getModel()
    reactions = model.getListOfReactions()
    for reaction in reactions:
        s_id = reaction.getId()
        klaw = reaction.getKineticLaw()
        klaw_math = klaw.getMath()
        print(formulaToString(klaw_math))
        # Relies on species; MathML doesn't type so you have to decide which
        # leaf nodes of the AST are species, params, etc.
    return

def sbml_to_species(document, species_df):
    model = document.getModel()
    species_list = model.getListOfSpecies()
    for species in species_list:
        s_id = species.getId()
        comp_name = species.getCompartment()
        u_def = species.getDerivedUnitDefinition()
        units = sbml_unit_definition_to_pint_quantity(u_def)
        init_cond = species.getInitialConcentration()
        # Convert to SI units
        init_cond *= units.magnitude
        group = species.getNotesString()

        species_df.append(s_id, group, units.units, init_cond, 10, (
                            unit.um ** 2) / unit.s, comp_name)
    return



sbml_to_compartments(doc, c_df)
sbml_to_parameters(doc, p_df)
# sbml_to_reactions(doc, r_df)
sbml_to_species(doc, s_df)

c_df.df.to_json(fp.split(".xml")[0] + "_compartments.json")
p_df.df.to_json(fp.split(".xml")[0] + "_parameters.json")
# r_df.df.to_json(fp.split(".xml")[0] + "_reactions.json")
s_df.df.to_json(fp.split(".xml")[0] + "_species.json")

