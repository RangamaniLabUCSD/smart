from libsbml import *
from sbml_unit_registry import SBMLUnit, SBML_UNIT_REGISTRY


import sys
import os
import os.path

import stubs
unit = stubs.unit

# taken from https://www.ebi.ac.uk/biomodels/BIOMD0000000932#Files
# fp = "models/Garde2020.xml"

# taken from https://www.ebi.ac.uk/biomodels/BIOMD0000000448#Files
# fp = "models/BIOMD0000000448_url.xml"

# taken from https://www.ebi.ac.uk/biomodels/BIOMD0000000931#Files
fp = "models/Voliotis2019.xml"

doc = readSBML(fp)
if doc.getNumErrors() > 0:
    raise Exception('Unable to read SBML from ' + fp)

c_df = stubs.model_building.CompartmentDF()

def sbml_to_compartment(document, compartment_df):
    marker = 1
    model = document.getModel()
    compartments = model.getListOfCompartments()
    for compartment in compartments:
        s_id = compartment.getId()
        units = compartment.getUnits()
        if units == "":
            units = "meter"
        dim = compartment.getSpatialDimensions()
        print(s_id, units, dim, marker)
        compartment_df.append(s_id, dim, units, marker)
        marker += 1
    return

sbml_to_compartment(doc, c_df)

c_df.df.to_json(fp.split(".xml")[0] + "_compartments.json")

