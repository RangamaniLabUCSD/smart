# import 
from libsbml import *

# Initialize overarching document.
document = SBMLDocument(3, 1) # Level 3, Version 1

# Initialize the model, without default units.
model = document.createModel()
#model.setTimeUnits('second')
#model.setExtentUnits('mole')
#model.setSubstanceUnits('mole')

# Define units. These are accessed by ID and are derived from toy_model.py
um_sq = model.createUnitDefinition()
um_sq.setId('micrometers_squared')
unit = um_sq.createUnit()
unit.setKind(UNIT_KIND_METRE)
unit.setExponent(2)
unit.setScale(-12)
unit.setMultiplier(1)

um_c = model.createUnitDefinition()
um_c.setId('micrometers_cubed')
unit = um_c.createUnit()
unit.setKind(UNIT_KIND_METRE)
unit.setExponent(3)
unit.setScale(-18)
unit.setMultiplier(1)

uM = model.createUnitDefinition()
uM.setId('micromoles')
unit = uM.createUnit()
unit.setKind(UNIT_KIND_MOLE)
unit.setExponent(1)
unit.setScale(-6)
unit.setMultiplier(1)

per_second = model.createUnitDefinition()
per_second.setId('per_second')
unit = per_second.createUnit()
unit.setKind(UNIT_KIND_SECOND)
unit.setExponent(-1)
unit.setScale(0)
unit.setMultiplier(1)

per_uM_second = model.createUnitDefinition()
per_uM_second.setId('per_uM_seconds')
unit = per_uM_second.createUnit()
unit.setKind(UNIT_KIND_SECOND)
unit.setExponent(-1)
unit.setScale(0)
unit.setMultiplier(6.022140857e-17)


# Define compartments.
cyto = model.createCompartment()
cyto.setId('cyto')
cyto.setConstant(True)
cyto.setSize(3)
cyto.setSpatialDimensions(3)
cyto.setUnits('micrometers_cubed')

pm = model.createCompartment()
pm.setId('pm')
pm.setConstant(True)
pm.setSize(2)
pm.setSpatialDimensions(2)
pm.setUnits('micrometers_squared')

# Define species.
s_a = model.createSpecies()
s_a.setId('A')
s_a.setCompartment('cyto')
s_a.setConstant(False)
s_a.setInitialAmount(10)
s_a.setSubstanceUnits('micromoles')

s_x = model.createSpecies()
s_x.setId('X')
s_x.setCompartment('pm')
s_x.setConstant(False)
s_x.setInitialAmount(1000)
s_x.setSubstanceUnits('item')

s_b = model.createSpecies()
s_b.setId('B')
s_b.setCompartment('pm')
s_b.setConstant(False)
s_b.setInitialAmount(0)
s_b.setSubstanceUnits('item')

# Define parameters.
kf = model.createParameter()
kf.setId('kf')
kf.setConstant(True)
kf.setValue(5)
kf.setUnits('per_uM_seconds')

kr = model.createParameter()
kr.setId('kr')
kr.setConstant(True)
kr.setValue(0.1)
kr.setUnits('per_second')

# Define reactions.
r = model.createReaction()
r.setId('A+X <-> B')
r.setReversible(True)
r.setFast(False)

# Attach reactants/products to reactions.
species_refA = r.createReactant()
species_refA.setSpecies('A')
species_refA.setConstant(False)

species_refX = r.createReactant()
species_refX.setSpecies('X')
species_refX.setConstant(False)

species_refB = r.createProduct()
species_refB.setSpecies('B')
species_refB.setConstant(False)

# Use MathML to define rates and attach.
# We can only attach one KineticLaw to one reaction, so we're just going to set
# the forward reaction and set reversible to true.
forward_math_ast = parseL3Formula('kf * ((A * cyto) + (X * pm))')
forward_k_law = r.createKineticLaw()
forward_k_law.setMath(forward_math_ast)

#reverse_math_ast = parseL3Formula('kr * B * pm')
#reverse_k_law = r.createKineticLaw()
#reverse_k_law.setMath(reverse_math_ast)

# Print final document and write to sbml.xml
sbml_str = writeSBMLToString(document)
print(sbml_str)

outfile = open('sbml.xml', 'w')
outfile.write(sbml_str)
outfile.close()

