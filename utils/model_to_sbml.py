from libsbml import *
from sbml_unit_registry import SBMLUnit, SBML_UNIT_REGISTRY

import sys
sys.path.append("../")

import stubs
unit = stubs.unit

c = stubs.model_building.CompartmentDF()

c.append('cyto', 3, unit.um, 1)

def compartment_to_sbml(compartment, model):
    sbml_comp = model.createCompartment()
    sbml_comp.setId(compartment.name)
    sbml_comp.setConstant(True)
    # sbml_comp.setSize(float(compartment.cell_marker))
    sbml_comp.setSpatialDimensions(int(compartment.dimensionality))
    sbml_comp.setUnits(compartment.compartment_units)
    return 

    # make sure units are in model, if not, add them
    # model_units = model.getListOfUnitDefinitions()
    # if model_units.getElementBySId(compartment.compartment_units) == None:
    #     SBML_UNIT_REGISTRY[compartment.compartment_units].add_to_model(model)

document = SBMLDocument(3, 1)
m = document.createModel()
# compartment_to_sbml(c.df['cyto'], m)
compartment_to_sbml(c.df.iloc[0, :], m)

# Print final document
sbml_str = writeSBMLToString(document)
print(sbml_str)
