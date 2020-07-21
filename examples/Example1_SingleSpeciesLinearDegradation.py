"""
Simple example showing linear degradation of a single species, A.
"""
import os
cwd=os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append("../../")
import stubs
unit = stubs.unit # unit registry

# initialize 
p = stubs.model_building.ParameterDF()
s = stubs.model_building.SpeciesDF()
c = stubs.model_building.CompartmentDF()
r = stubs.model_building.ReactionDF()

### define parameters
# name, value, unit, notes
p.append('kdeg', 5.0, 1/(unit.s), 'degradation rate')

### define species
# name, plot group, concentration units, initial condition, diffusion
# coefficient, diffusion coefficient units, compartment
s.append('A', 'cytosolic', unit.uM, 10, 1, unit.um**2/unit.s, 'cyto')

### define compartments
# name, geometric dimensionality, length scale units, marker value
c.append('cyto', 3, unit.um, 1)

### define reactions
# name, notes, left hand side of reaction, right hand side of reaction, kinetic
# parameters
r.append('A linear degredation', 'example reaction', ['A'], [], {"on": "kdeg"}, reaction_type='mass_action_forward')


# write out to file
stubs.common.write_smodel(cwd + '/model.smodel', p, s, c, r)