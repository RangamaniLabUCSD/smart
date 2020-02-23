"""
Simple example showing diffusion of a molecule A binding to molecule X on the
membrane to create molecule B.
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
p.append('kf', 5.0, 1/(unit.uM*unit.s), 'forward rate')
p.append('kr', 0.1, 1/unit.s, 'reverse rate')

### define species
# name, plot group, concentration units, initial condition, diffusion
# coefficient, diffusion coefficient units, compartment
s.append('A', 'cytosolic', unit.uM, 10.0, 10, unit.um**2/unit.s, 'cyto')
s.append('X', 'membrane bound', unit.molecule/unit.um**2, 1000.0, 0.1, unit.um**2/unit.s, 'pm')
s.append('B', 'membrane bound', unit.molecule/unit.um**2, 0.0, 50, unit.um**2/unit.s, 'pm')

### define compartments
# name, geometric dimensionality, length scale units, marker value
c.append('cyto', 3, unit.um, 1)
c.append('pm', 2, unit.um, 2)

### define reactions
# name, notes, left hand side of reaction, right hand side of reaction, kinetic
# parameters
r.append('A+X <-> B', 'cell 2013', ['A','X'], ['B'], {"on": "kf", "off": "kr"})


# write out to file
p.write_json(name=cwd + '/parameters.json')
s.write_json(name=cwd + '/species.json')
c.write_json(name=cwd + '/compartments.json')
r.write_json(name=cwd + '/reactions.json')
