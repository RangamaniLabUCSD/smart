from libsbml import *

class SBMLUnit:
    def __init__(self, id, kind, exponent, scale, multiplier):
        self.id = id
        self.kind = kind
        self.exponent = exponent
        self.scale = scale
        self.multiplier = multiplier

    def add_to_model(self, model):
        ud = model.createUnitDefinition()
        ud.setId(self.id)
        u = ud.createUnit()
        u.setKind(self.kind)
        u.setExponent(self.exponent)
        u.setMultiplier(self.multiplier)

SBML_UNIT_REGISTRY = {
    # "micrometers_squared" : SBMLUnit('micrometer', UNIT_KIND_METRE, 2, -12, 1)
    "micrometer" : SBMLUnit('micrometer', UNIT_KIND_METRE, 1, -6, 1)
}