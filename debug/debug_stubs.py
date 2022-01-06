"""Scratch code for whatever I'm trying to figure out at the moment"""

import stubs
import stubs.common as common
import dolfin as d
import numpy as np
import pint
unit = stubs.unit # unit registry

# #======

# aliases - unit registry
from stubs.model_assembly import Parameter, Species, Compartment, Reaction
uM       = unit.uM
meter    = unit.m
um       = unit.um
molecule = unit.molecule
sec      = unit.s

pc, sc, cc, rc = common.empty_sbmodel()

# parameters
pc.add([    
    #Parameter('kf'      , 5.0, meter/sec, 'forward rate'),
    # volume-to-volume [m/s]
    Parameter.from_expression('gating_f' , '5.0+t', um/sec),
    Parameter('gating_r'      , 1.0, um/sec),#um/sec, 'reverse rate'),
    # volume mass-action 2 to 1
    Parameter('kf', 3.0, 1/(uM*sec), 'volume mass-action forward A+B -> C'),
    Parameter('kr', 7.0, 1/(sec), 'volume mass-action reverse C -> A+B'),
    # volume to surface / surface to volume
    Parameter('kf_AX_X2', 3.0, 1/(uM*sec), 'A+X -> X2'),
    Parameter('kr_AX_X2', 3.0, 1/(sec), 'X2 -> A+X'),
    # volume-surface to volume 
    Parameter('kf_AY_B', 3.0, 1/(uM*sec), 'A+Y -> B'),
    # volume-volume to surface 
    Parameter('kf_AB_Y', 3.0, um/(uM*sec), 'A+B -> Y'),
    # volume/surface degradation [1/s]
    Parameter('kdeg_B', 2.0, 1/sec, 'degradation rate'),
    Parameter('kdeg_X', 2.0, 1/sec, 'degradation rate'),
    Parameter('kdeg_Y', 2.0, 1/sec, 'degradation rate'),
])

# species
sc.add([
    #Species('A'   , 10 , uM            , 100, um**2/sec, 'cytosol'),
    Species('A'    , 'z'  , uM            , 100, um**2/sec, 'cytosol'),
    Species('A2'   , 'z+3', uM            , 100, um**2/sec, 'cytosol'),
    Species('A3'   , 0    , uM            , 100, um**2/sec, 'cytosol'),
    Species('B' , 3    , uM            , 100, um**2/sec, 'er_vol'),
    Species('X' , 100  , molecule/um**2, 10 , um**2/sec, 'pm'),
    Species('X2' , 40  , molecule/um**2, 10 , um**2/sec, 'pm'),
    Species('Y', 60  , molecule/um**2, 10 , um**2/sec, 'er_mem'),
])

# compartments
cc.add([
    Compartment('cytosol', 3, um, 11),
    Compartment('er_vol' , 3, um, 12),
    #Compartment('test_vol' , 3, um, [3,5]),
    Compartment('pm'     , 2, um, 2),
    Compartment('er_mem' , 2, um, 4),
])

# flux topologies are commented
rc.add([
    Reaction('A <-> B'      , ['A']     , ['B'] , {'on': 'gating_f', 'off': 'gating_r'} , explicit_restriction_to_domain='er_mem'), # [volume_to_volume] 
    Reaction('A + A2 <-> A3', ['A','A2'], ['A3'], {'on': 'kf',       'off': 'kr'}                                                ), # [volume] volume mass action (2 to 1)
    Reaction('B -> 0'       , ['B']     , []    , {'on': 'kdeg_B'}                      , reaction_type='mass_action_forward'    ), # [volume] degradation
    Reaction('A + X <-> X2' , ['A','X'] , ['X2'], {'on': 'kf_AX_X2', 'off': 'kr_AX_X2'}                                          ), # [volume_to_surface] [surface_to_volume]
    Reaction('X -> 0'       , ['X']     , []    , {'on': 'kdeg_X'}                      , reaction_type='mass_action_forward'    ), # [surface] degradation
    Reaction('Y -> 0'       , ['Y']     , []    , {'on': 'kdeg_Y'}                      , reaction_type='mass_action_forward'    ), # [surface] degradation
    Reaction('A + Y <-> B'  , ['A','Y'] , ['B'] , {'on': 'kf_AY_B'}                     , reaction_type='mass_action_forward'    ), # [volume-surface_to_volume]
    Reaction('A + B <-> Y'  , ['A','B'] , ['Y'] , {'on': 'kf_AB_Y'}                     , reaction_type='mass_action_forward'    ), # [volume-volume_to_surface]
])

# config
stubs_config = stubs.config.Config()
# stubs_config.loglevel['dolfin'] = 'CRITICAL'
# Define solvers
mps           = stubs.solvers.MultiphysicsSolver()
nls           = stubs.solvers.NonlinearNewtonSolver()
ls            = stubs.solvers.DolfinKrylovSolver()
solver_system = stubs.solvers.SolverSystem(final_t=0.1, initial_dt=0.01, multiphysics_solver=mps, nonlinear_solver=nls, linear_solver=ls)
# mesh
from pathlib import Path
path    = Path('.').resolve()
subdir  = 'data'
while True:
    if path.parts[-1]=='stubs' and path.joinpath(subdir).is_dir():
        path = path.joinpath(subdir)
        break
    path = path.parent
stubs_mesh = stubs.mesh.ParentMesh(mesh_filename=str(path / 'adjacent_cubes_refined.xml'))
#stubs_mesh.dolfin_mesh = d.refine(stubs_mesh.dolfin_mesh)

model = stubs.model.Model(pc, sc, cc, rc, stubs_config, solver_system, stubs_mesh)

# aliases
p = model.pc.get_index(0)
s = model.sc.get_index(0)
s2 = model.sc.get_index(2)
c = model.cc.get_index(0)
r = model.rc.get_index(0)
r2 = model.rc.get_index(2)

import cProfile

model._init_1()
model._init_2()
cProfile.run("model._init_3()")                 
model._init_4()                 
model._init_5_1_reactions_to_fluxes()
model._init_5_2_set_flux_units()
#model._init_5_3_reaction_fluxes_to_forms()

#aliases
m = model.parent_mesh
u = s.u['u']
f = model.fc.get_index(13)
f1 = model.fc.get_index(1)
r = f.reaction

# aliases
A = model.sc['A']
B = model.sc['B']
Y = model.sc['Y']
X = model.sc['X']
mtot  = model.parent_mesh
mcyto = model.cc['cytosol'].mesh
merv  = model.cc['er_vol'].mesh
merm  = model.cc['er_mem'].mesh
mpm   = model.cc['pm'].mesh

mall = [mtot, mcyto, merv, merm, mpm]

# build mappings (dim 2 -> 3)
# merm.dolfin_mesh.build_mapping(mcyto.dolfin_mesh)
# merm.dolfin_mesh.build_mapping(merv.dolfin_mesh)


#mpm.dolfin_mesh.build_mapping(merv.dolfin_mesh)

# vol to vol
# d.assemble((A.u['u'] - B.u['u']) * merm.dx)

# surf to vol
#print(d.assemble((A.u['u'] * X_pm.u['u']) * mpm.dx_map[mcyto.id](1))) 
print(d.assemble((A.u['u']) * mpm.dx_map[mcyto.id](1))) 
print(f"expected value: 24")

#d.assemble((A.u['u'] * X_pm.u['u']) * mcyto.ds(2)) # not implemented yet

#merm.mesh_view


# for m in mall:
#     #print(m.mesh_view)
#     print(m.id)

# build mappings (dim 2 -> 3)
# merm.dolfin_mesh.build_mapping(mcyto.dolfin_mesh)
# merm.dolfin_mesh.build_mapping(merv.dolfin_mesh)
# mpm.dolfin_mesh.build_mapping(mcyto.dolfin_mesh)
# mpm.dolfin_mesh.build_mapping(merv.dolfin_mesh)

# mtest = d.MeshView.create(mcyto.mf['facets'], 4)
# mcyto.mesh_view[3].create(mcyto.mf['facets'], 4)








# #==============================
# #==============================
# # MWE volume-volume
# #==============================
# #==============================
# m = d.Mesh('data/adjacent_cubes.xml')
# mf3 = d.MeshFunction('size_t', m, 3, value=m.domains())
# mf2 = d.MeshFunction('size_t', m, 2, value=m.domains())

# # try unmarking some of mf2(4)
# d.CompiledSubDomain('x[0] < 0.0').mark(mf2,6)

# m1 = d.MeshView.create(mf3, 11)
# m2 = d.MeshView.create(mf3, 12)
# m_ = d.MeshView.create(mf2, 4)

# V0 = d.FunctionSpace(m,"P",1); V1 = d.FunctionSpace(m1,"P",1); V2 = d.FunctionSpace(m2,"P",1); V_ = d.FunctionSpace(m_,"P",1);
# V  = d.MixedFunctionSpace(V0,V1,V2,V_)

# u = d.Function(V)
# u0 = u.sub(0); u1 = u.sub(1); u2 = u.sub(2); u_ = u.sub(3)
# u0.vector()[:] = 1; u1.vector()[:] = 5; u2.vector()[:] = 3; u_.vector()[:] = 7

# (ut0, ut1, ut2, ut_) = d.TrialFunctions(V)


# v = d.TestFunctions(V)
# v0,v1,v2,v_ = v

# u = d.Function(V)

# dx0 = d.Measure('dx', domain=m);  dx1 = d.Measure('dx', domain=m1);
# dx2 = d.Measure('dx', domain=m2); dx_ = d.Measure('dx', domain=m_);

# mf1 = d.MeshFunction('size_t', m1, 2, 0)
# d.CompiledSubDomain('near(x[2],0)').mark(mf1,4)
# ds1 = d.Measure('ds', domain=V1.mesh(), subdomain_data=mf1)

# m_.build_mapping(m1)
# m_.build_mapping(m2)

# # volume PDE: volume+surface -> volume
# def volPDE_volsurf2vol():
#     return d.assemble(u1*u_*u2*v1*dx_)

# # volume PDE: volume -> volume
# def volPDE_vol2vol():
#     # m1.build_mapping(m2) does not work
#     # needs mapping
#     return d.assemble((u1-u2)*v1*dx_)

# # surface PDE: volume -> surface
# # only works after m_.build_mapping(m1)
# def surfPDE_vol2surf_trial():
#     return d.assemble(ut1*v_*dx_) #Works

# def surfPDE_vol2surf():
#     return d.assemble(u1*v_*dx_) #Segfault if mapping is not built

# # volume PDE: surface -> volume
# def volPDE_surf2vol_trial():
#     return d.assemble(ut_*v1*dx_) #Works

# def volPDE_surf2vol():
#     return d.assemble(u_*v1*dx_) #Works

# # volume PDE: volume -> surface
# def volPDE_vol2surf_trial():
#     return d.assemble(ut1*v1*dx_) #Works

# def volPDE_vol2surf():
#     return d.assemble(u1*v1*dx_) #Segfault if mapping is not built


# # d.assemble(u_*v1*ds1) does not work (Exception: codim != 0 or 1 - Not (yet) implemented)

# # volume -> volume
