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

print(d.assemble_mixed(model.Fblocks[0][0]).get_local().mean())
print(d.assemble_mixed(model.Fblocks[0][1]).get_local().mean())
# 0.013009917771822541
# 0.4652419890515132
# 0.012918298632584353
# 0.4652419890515132
#====================== 02/07/2022
# from ufl.algorithms.ad import expand_derivatives
# from ufl.form import sub_forms_by_domain

# ermem = model.child_meshes['er_mem']

# dx = ermem.dx
# f = (A.u['u'] - B.u['u']) * model.v[0][0] * dx
# F = d.assemble_mixed(f)
# dfdu = sub_forms_by_domain(expand_derivatives(d.derivative(f, model.u['u'].sub(1))))
# M = d.assemble_mixed(dfdu[0])

# dxnew = ermem.intersection_dx[frozenset({17, 23})]
# fnew = (A.u['u'] - B.u['u']) * model.v[0][0] * dxnew
# Fnew = d.assemble_mixed(fnew)
# dfnewdu = sub_forms_by_domain(expand_derivatives(d.derivative(fnew, model.u['u'].sub(1))))
# Mnew = d.assemble_mixed(dfnewdu[0])

# np.all(F.get_local() == Fnew.get_local())

# # M.nnz()
# # Mnew.nnz()
# # M.array().mean()
# # Mnew.array().mean()
# np.all(M.array() == Mnew.array())

# M_ = d.assemble_mixed(model.Jblocks[1][0])
# np.all(M.array() == M_.array())

#Fblock, Jblock, _ = model.get_block_system(f, model.u['u']._functions)



#d.derivative(model.Fsum, model.u['u'].sub(1))
# J = model.problem.Jpetsc_nest.getNestSubMatrix(0,1)
# M = d.assemble_mixed(model.Jblocks[1][0])

# # compare nonzeros
# print(M.nnz())
# print(J.getInfo()['nz_used'])
# # compare size
# print((M.size(0), M.size(1)))
# print(J.size)

# M2 = d.as_backend_type(d.assemble_mixed(model.Jblocks[1][0], tensor=d.PETScMatrix())).mat()
# Jpetsc = []
# Jsum=None
# Jsum = d.as_backend_type(d.assemble_mixed(model.Jblocks[1][0], tensor=d.PETScMatrix()))
# Jpetsc.append(Jsum)

#Jblocks[1][0] is dFcyto / duervol



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



# pre:
# species A : ((10.0, 12.0)), mass=88.00000000000874
# species A2 : ((5.0, 7.0)), mass=47.99999999999816
# species A3 : ((7.0, 9.0)), mass=63.99999999999438
# species B : ((3.0, 3.0)), mass=24.00000000000639
# species X : ((98.0, 102.0)), mass=3592.000000000033
# species X2 : ((40.0, 40.0)), mass=1440.000000000049

# solve finished in 3.975677 seconds ...................................................................................
# post:
# species A : ((7.481665301596026, 8.096608205553615)), mass=62.769263576269694
# species A2 : ((2.9123836951791344, 3.2444901947820486)), mass=24.61705431501025
# species A3 : ((10.603370173101725, 11.239755936937092)), mass=87.38294568498942
# species B : ((3.0933085739651727, 3.3565925682036064)), mass=25.370635544103223
# species X : ((83.23650610051348, 98.64883200350206)), mass=3301.858132382143
# species X2 : ((40.002242679047306, 58.3385855739414)), mass=1730.141867617874
# 2.9123836951791344
# 11.239755936937092
# 62.769263576269694




# print(f"sum of residuals {sum(model.get_total_residual())}")
# # 743.7527142501965
# # 1e-6

# # ===================
# pm = model.child_meshes['pm']
# pm_cyto = pm.intersection_submesh[frozenset({18})]
# model.Jblocks[5][0] # ervol -> pm # 

# model.Jblocks[0][0] # d(Fcyto)/d(ucyto) (domain=pm_intersect_cytosol) -> pm #  Mesh entity index -1 out of range [0, 49600] for entity of dimension 2.
# # A ucyto * X upm (domain = pm_intersect_cytosol)
# form = model.u['u'].sub(0).sub(0) * model.u['u'].sub(2).sub(0) * model.v[0][0] * pm.intersection_dx[frozenset({18})]
# F = d.assemble_mixed(form)
# dFdu = d.derivative(form, model.u['u'].sub(0).sub(0))
#  #5 │                2 │   54 │ pm_intersect_cytosol            │        1600 │         2480 │            880
# # M = d.assemble(model.Jblocks[0][0])
# # M.array()[0:10,0:10].sum()  # old dolfin = 0.1466, stubs dolfin = 0.1745
# # M.nnz()                     # old dolfin = 113877, stubs dolfin = 113814
# # #M.size(0) = 14553

# # import petsc4py.PETSc as petsc
# # # create a petsc.Mat of zeros with size (10,10)
# # # M = petsc.Mat().create(petsc.COMM_WORLD)
# # # M.setSizes((10,10))


# flatten = stubs.model_assembly.flatten
# jblocks = flatten(model.Jblocks)
# idx = 0
# for j in jblocks:
#     print(f"{idx}: {j.num_coefficients()}")
#     print(type(j.function_space(0)))
#     d.assemble_mixed(j)
#     idx +=1

#     # try to get the mesh from j (type dolfin.cpp.fem.Form)
# try:
#     mesh = j0.mesh()
# except:
#     pass

# # f = d.cpp.fem.Form(2,0)
# #f.set_mesh
# # d.assemble_mixed(f)


# # new meshview
# pm_to_parent = pm.mesh_view[pm.parent_mesh.id].cell_map()
# pm_ervol_intersecting = pm.intersection_map[frozenset({24})]
# # get indices of pm_ervol_intersecting where equal to 1
# indices = np.where(pm_ervol_intersecting.array() == 1)[0]
# parent_indices = np.array(pm_to_parent)[indices]
# # new mesh function
# mf2 = d.MeshFunction('size_t', pm.parent_mesh.dolfin_mesh, 2, value=0)
# mf2.array()[parent_indices] = 1

# pm_ervol_mesh = d.MeshView.create(mf2, 1)
# dx = d.Measure('dx', pm_ervol_mesh)


# # test
# form = model.u['u'].sub(1) * model.u['u'].sub(2).sub(0) * model.v[1] * pm.dx_map[frozenset({24})](1)
# form = model.u['u'].sub(1) * model.u['u'].sub(2).sub(0) * model.v[1] * pm.intersection_dx[frozenset({24})]
# F2 = d.assemble_mixed(form)
# M2 = d.assemble_mixed(d.derivative(form, model.u['u'].sub(1)))
# M2.array().max() # old dolfin = 223.233, stubs dolfin = 0.998
# F2.max()         # old dolfin = 1380.41, stubs dolfin = 11.988

# # integrate B*X2 over region where pm intersects with ervol [meshid=24] (area= 20) (total pm area=36)
# form = model.u['u'].sub(1) * model.u['u'].sub(2).sub(1) * pm.dx_map[frozenset({24})](1)
# form = model.u['u'].sub(1) * model.u['u'].sub(2).sub(1) * pm.intersection_dx[frozenset({24})]
# # B*X  -> old dolfin = 9493.56, stubs dolfin = 5928.0
# # B*X2 -> old dolfin = 6250.87, stubs dolfin = 2400, analytic = 3 * 40 * 16 = 2400
# d.assemble_mixed(form) 

# # newform = model.u['u'].sub(1) * model.u['u'].sub(2).sub(0) * model.v[1] * dx
# # newF = d.assemble_mixed(newform)
# # d.assemble_mixed(d.derivative(newform, model.u['u'].sub(1)))


# # compare
# print(F.get_local().mean())
# print(newF.get_local().mean())
# print(F.get_local().var())
# print(newF.get_local().var())




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