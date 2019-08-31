import os
cwd=os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append("../../")
import fsbgen as fsb
unit = fsb.unit # unit registry

# initialize 
p = fsb.model_building.ParameterDF()
s = fsb.model_building.SpeciesDF()
c = fsb.model_building.CompartmentDF()
r = fsb.model_building.ReactionDF()

# parameters
p.append('kf', 1.0, 1/(unit.uM*unit.s), 'cell 2013')
p.append('kr', 0.1, 1/unit.s, 'cell 2013')
p.append('kf_buff', 1.0, 1/(unit.uM*unit.s), 'a buffer')
p.append('kr_buff', 0.1, 1/unit.s, 'a buffer')

# species
s.append('A', 'cell 2013', unit.uM, 0.0, 10, unit.um**2/unit.s, 'cyto')
s.append('X', 'cell 2013', unit.molecule/unit.um**2, 1000.0, 0.1, unit.um**2/unit.s, 'pm')
s.append('B', 'cell 2013', unit.molecule/unit.um**2, 1.0, 50, unit.um**2/unit.s, 'pm')
#s.append('C', 'a buffer', unit.uM, 1.0, 50, unit.um**2/unit.s, 'cyto')
#s.append('buff', 'a buffer', unit.uM, 1.0, 50, unit.um**2/unit.s, 'cyto')


# compartments
c.append('cyto', 3, unit.um, 1)
c.append('pm', 2, unit.um, 2)

# reactions
r.append('A+X <-> B', 'cell 2013', ['A','X'], ['B'], {"on": "kf", "off": "kr"})
#r.append('A + buff <-> C', 'a buffer', ['A', 'buff'], ['C'], {"on": 'kf_buff', "off": "kr_buff"})


# write out to file
p.write_json(name=cwd + '/parameters.json')
s.write_json(name=cwd + '/species.json')
c.write_json(name=cwd + '/compartments.json')
r.write_json(name=cwd + '/reactions.json')

