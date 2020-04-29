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
    UNIT_KIND_DIMENSIONLESS: 1,

    # pint to SBML
    '[time]': UNIT_KIND_SECOND,
    '[length]': UNIT_KIND_METRE,
    '[substance]': UNIT_KIND_MOLE
}

def convert_sbml_unit_to_pint(sbml_unit):
    kind = sbml_unit.kind
    scale = sbml_unit.scale
    exp = sbml_unit.exponent
    mult = sbml_unit.multiplier
    return (mult * SBML_BASE_UNIT_REGISTRY[kind]) ** exp * (10 ** scale)

def convert_pint_to_sbml_unit_definition(pint_unit, model):
    u_def = model.createUnitDefinition()
    u_def.setId(pint_unit.format_babel())

    quant = 1 * pint_unit
    quant = quant.to_base_units()
    magnitude = quant.magnitude
    unit_dict = quant.units.dimensionality.items()
    first_unit = True

    for unit_type, dimension in unit_dict:
        unit = u_def.createUnit()
        unit.setKind(SBML_BASE_UNIT_REGISTRY[unit_type])
        unit.setExponent(dimension)
        if first_unit:
            unit.setMultiplier(float(magnitude))
            first_unit = False

    return

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

doc = SBMLDocument(3, 1)
model = doc.createModel()
u_a = 5 * unit_registry.kilometers ** 3
u_b = 10 * unit_registry.molecules / unit_registry.miles ** 3
u_c = 5 * unit_registry.inches / unit_registry.hours
convert_pint_to_sbml_unit_definition(u_a.units, model)
convert_pint_to_sbml_unit_definition(u_b.units, model)
convert_pint_to_sbml_unit_definition(u_c.units, model)
print(writeSBMLToString(doc))

# fp = "models/Voliotis2019.xml"
# doc = readSBML(fp)
# print(extract_units_from_sbml_document(doc))
