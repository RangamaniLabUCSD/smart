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
p.append('kdeg', 2.0, 1/(unit.s), 'degradation rate')

### define species
# name, plot group, concentration units, initial condition, diffusion
# coefficient, diffusion coefficient units, compartment
s.append('A', 'cytosolic', unit.molecule/unit.um**3, 10.0, 10, unit.um**2/unit.s, 'cyto')
s.append('X', 'membrane bound', unit.molecule/unit.um**2, 1000.0, 0.1, unit.um**2/unit.s, 'pm')
s.append('B', 'membrane bound', unit.molecule/unit.um**2, 0.0, 50, unit.um**2/unit.s, 'pm')

### define compartments
# name, geometric dimensionality, length scale units, marker value
c.append('cyto', 3, unit.um, 1)
c.append('pm', 2, unit.um, 2)

### define reactions
# name, notes, left hand side of reaction, right hand side of reaction, kinetic
# parameters
r.append('A+X <-> B', 'First reaction', ['A','X'], ['B'], {"on": "kf", "off": "kr"})
r.append('B linear degredation', 'second reaction', ['B'], [], {"on": "kdeg"}, reaction_type='mass_action_forward')




# Sample

# define parameters
parameter(name='kf', value=1.0, units=1/(unit.uM*unit.s))
parameter(name='kr', value=0.1, units=1/unit.s)

# define compartments
compartment(name='cyto', geom_dim=3, units=unit.um, marker_value=1)
compartment(name='pm',   geom_dim=2, units=unit.um, marker_value=2)

# define species
species(name='A', init_value=1.0,    units=unit.uM,                  D=1.0,  D_units=unit.um**2/unit.s, compartment='cyto')
species(name='X', init_value=1000.0, units=unit.molecule/unit.um**2, D=0.1,  D_units=unit.um**2/unit.s, compartment='pm')
species(name='B', init_value=0.0,    units=unit.molecule/unit.um**2, D=0.01, D_units=unit.um**2/unit.s, compartment='pm')

# define reactions
reaction(name='A+X <-> B', LHS=['A','X'], RHS=['B'], parameters={"on": "kf", "off": "kr"}, reaction_type='mass_action')


# write out to file
stubs.common.write_smodel(cwd + '/cell2013_3d.smodel', p, s, c, r)
