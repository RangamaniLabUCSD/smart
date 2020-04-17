from libsbml import *

import stubs
unit_registry = stubs.unit

SBML_BASE_UNIT_REGISTRY = {
    # SBML to pint
    UNIT_KIND_SECOND: unit_registry.second,
    UNIT_KIND_METRE: unit_registry.meter,
    UNIT_KIND_MOLE: unit_registry.mole,
    UNIT_KIND_VOLT: unit_registry.volt,
    UNIT_KIND_LITRE: unit_registry.liter,
    UNIT_KIND_ITEM: unit_registry.molecule,
    UNIT_KIND_DIMENSIONLESS: 1

    # pint to SBML
}

def convert_sbml_unit_to_pint(sbml_unit):
    kind = sbml_unit.kind
    scale = sbml_unit.scale
    exp = sbml_unit.exponent
    mult = sbml_unit.multiplier
    return (mult * SBML_BASE_UNIT_REGISTRY[kind]) ** exp * (10 ** scale)

def extract_units_from_sbml_document(document):
    if document.getNumErrors() > 0:
        raise Exception("Errors detected while reading document.")
    return extract_units_from_sbml(document.getModel())

def extract_units_from_sbml(model):
    unit_defs = model.getListOfUnitDefinitions()
    pint_units = {}

    for unit_def in unit_defs:
        sid = unit_def.id
        pint_unit = 1
        units = unit_def.getListOfUnits()
        for unit in units:
            pint_unit *= convert_sbml_unit_to_pint(unit)
        pint_units[sid] = pint_unit

    return pint_units

fp = "models/Voliotis2019.xml"
doc = readSBML(fp)
print(extract_units_from_sbml_document(doc))
