"""Scratch code for whatever I'm trying to figure out at the moment"""
import stubs
import stubs.common as common
import dolfin as d
import ufl
import numpy as np
import pint
import logging
import itertools
unit = stubs.unit # unit registry

# ===================
# MPI
# ===================
rank = d.MPI.comm_world.rank

# ===================
# aliases - unit registry
# ===================
from stubs.model_assembly import Parameter, Species, Compartment, Reaction
uM       = unit.uM
meter    = unit.m
um       = unit.um
molecule = unit.molecule
sec      = unit.s

import stubs_model 

model = stubs_model.make_model('adjacent_cubes_from_dolfin_20_lessPM.h5')

# ===================
# logging
# ===================
# set for others
loglevel = 'INFO'
model.config.loglevel = {'FFC': loglevel,
                         'UFL': loglevel,
                         'dijitso': loglevel,
                         'dolfin': loglevel}
model.config.set_logger_levels()

# model.parent_mesh = stubs.mesh.ParentMesh(str(stubs.common.data_path() / 'adjacent_cubes_from_dolfin_40.h5'), 'hdf5')
print(model.parent_mesh.num_vertices)


use_snes = True
model.config.solver['use_snes'] = use_snes

# #====================
# # init model
# # ===================
# #import cProfile
#cProfile.run("model._init_3()")                 
model._init_1()
model._init_2()
model._init_3()
model._init_4()                 
model._init_5_1_reactions_to_fluxes()
model._init_5_2_create_variational_forms()
model._init_5_3_setup_variational_problem_and_solver()
A = model.sc['A']
A2 = model.sc['A2']
A3 = model.sc['A3']
B = model.sc['B']

print(model.u['u'].sub(0).compute_vertex_values().min())
print(model.u['u'].sub(0).compute_vertex_values().max())

print('pre: ')
for species in model.sc.values:
    umin = model.dolfin_get_function_values(species).min()
    umax = model.dolfin_get_function_values(species).max()
    mass = model.get_mass(species)
    print(f"species {species.name} : ({umin, umax}), mass={mass}")
model.stopwatch('solve')
if use_snes:
    model.solver.solve(None, model._ubackend)
else:
    model.solver.solve()
model.stopwatch('solve', stop=True)
print('post: ')
for species in model.sc.values:
    umin = model.dolfin_get_function_values(species).min()
    umax = model.dolfin_get_function_values(species).max()
    mass = model.get_mass(species)
    print(f"species {species.name} : ({umin, umax}), mass={mass}")

print(model.u['u'].sub(0).compute_vertex_values().min())
print(model.u['u'].sub(0).compute_vertex_values().max())

print(model.get_mass(A))

print(f"sum of residuals {sum(model.get_total_residual())}")
# 743.7527142501965
# 1e-6


#66.0937846550966

# pre:
# species A : ((10.0, 12.0)), mass=88.00000000000044
# species A2 : ((5.0, 7.0)), mass=48.0000000000001
# species A3 : ((7.0, 9.0)), mass=63.99999999999959
# species B : ((3.0, 3.0)), mass=23.99999999999995
# Solving mixed nonlinear variational problem.
#   Newton iteration 0: r (abs) = 5.444e+02 (tol = 1.000e-10) r (rel) = 1.000e+00 (tol = 1.000e-09)
#   Newton iteration 1: r (abs) = 5.978e+01 (tol = 1.000e-10) r (rel) = 1.098e-01 (tol = 1.000e-09)
#   Newton iteration 2: r (abs) = 1.077e+00 (tol = 1.000e-10) r (rel) = 1.978e-03 (tol = 1.000e-09)
#   Newton iteration 3: r (abs) = 4.883e-03 (tol = 1.000e-10) r (rel) = 8.970e-06 (tol = 1.000e-09)
#   Newton iteration 4: r (abs) = 2.270e-04 (tol = 1.000e-10) r (rel) = 4.170e-07 (tol = 1.000e-09)
#   Newton iteration 5: r (abs) = 1.054e-05 (tol = 1.000e-10) r (rel) = 1.936e-08 (tol = 1.000e-09)
#   Newton iteration 6: r (abs) = 4.895e-07 (tol = 1.000e-10) r (rel) = 8.991e-10 (tol = 1.000e-09)
#   Newton solver finished in 6 iterations and 6 linear solver iterations.
# post:
# species A : ((7.548343292705323, 8.14060106376936)), mass=63.16422134119935
# species A2 : ((2.9026068866967845, 3.2340659382693833)), mass=24.539835437570275
# species A3 : ((10.613295946765282, 11.249856899995862)), mass=87.46016456242998
# species B : ((3.0936829244406403, 3.3572813784050592)), mass=25.37599643259512






# # Time loop
# while True:
#     end_simulation = model.solve_single_timestep(plot_period)
#     if end_simulation:
#         break



# 2.9026068866967845
# 11.249856899995862
# 2.9026068866967845
# 11.249856899995862


#====================
# aliases
# ===================
# p = model.pc.get_index(0)
# s = model.sc.get_index(0)
# c = model.cc.get_index(0)
# r = model.rc.get_index(0)
# A = model.sc['A']
# B = model.sc['B']
# # Y = model.sc['Y']
# # X = model.sc['X']
# # mtot  = model.parent_mesh
# # mcyto = model.cc['cytosol'].mesh
# # merv  = model.cc['er_vol'].mesh
# # merm  = model.cc['er_mem'].mesh
# # mpm   = model.cc['pm'].mesh

# # by compartment
# F1 = Fbloc[0]
# #F11 = sum([f.lhs for f in model.forms if f.species.name == 'A'])
# F2 = Fbloc[1]
# u1 = u._functions[0]
# u2 = u._functions[1]
# u11 = A._usplit['u']

# cytoJ = Js[0:4]
# pmJ   = Js[4:8]
# ervJ  = Js[8:12] #nonzero
# ermJ  = Js[12:16] #nonzero