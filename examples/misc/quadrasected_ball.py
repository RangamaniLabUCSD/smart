"""
Bulk-Surface-Bulk Reaction Diffusion problem
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
p.append('kf', 1.0, unit.um/unit.s, 'forward rate')

### define species
# name, plot group, concentration units, initial condition, diffusion
# coefficient, diffusion coefficient units, compartment
s.append('A', 'cytosolic'   , unit.molecule/unit.um**3, 1.0, 1.0, unit.um**2/unit.s, 'cyto')
s.append('X', 'er membrane' , unit.molecule/unit.um**2, 0.5, 0.1, unit.um**2/unit.s, 'erm')
s.append('B', 'er'          , unit.molecule/unit.um**3, 5.0, 1.0, unit.um**2/unit.s, 'er')

### define compartments
# name, geometric dimensionality, length scale units, marker value
c.append('cyto' , 3, unit.um, 1)
c.append('erm'  , 2, unit.um, 4)
c.append('er'   , 3, unit.um, 3)

### define reactions
# name, notes, left hand side of reaction, right hand side of reaction, kinetic
# parameters
r.append('B <-X-> A', 'Leak from ER to cyto', ['B'], ['A'], {"perm": "kf"},
         reaction_type='leak_dynamic', species_map={'uhigh': 'B', 'ulow': 'A'})

#r.append('B linear degredation', 'second reaction', ['B'], [], {"on": "kdeg"}, reaction_type='mass_action_forward')



# write out to file
stubs.common.write_sbmodel(cwd + '/cell2013_3d.sbmodel', p, s, c, r)
