"""Scratch code for whatever I'm trying to figure out at the moment"""

import stubs
import stubs.common as common
import dolfin as d
import ufl
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
    Parameter('kf'      , 5.0, meter/sec, 'forward rate'),
    # # volume-to-volume [m/s]
    # Parameter('testing', 5, 1/sec),
    Parameter.from_expression('gating_f' , '5.0+t', um/sec),
    Parameter('gating_r'      , 1.0, um/sec),#um/sec, 'reverse rate'),
    # volume mass-action 2 to 1
    Parameter('kf', 13.0, 1/(uM*sec), 'volume mass-action forward A+A2 -> A3'),
    Parameter('kr', 2.0, 1/(sec), 'volume mass-action reverse A3 -> A+A2'),
    # # volume to surface / surface to volume
    Parameter('kf_AX_X2', 3.0, 1/(uM*sec), 'A+X -> X2'),
    Parameter('kr_AX_X2', 3.0, 1/(sec), 'X2 -> A+X'),
    # # volume-surface to volume 
    Parameter('kf_AY_B', 3.0, 1/(uM*sec), 'A+Y -> B'),
    # # volume-volume to surface 
    Parameter('kf_AB_Y', 3.0, um/(uM*sec), 'A+B -> Y'),
    # # volume/surface degradation [1/s]
    Parameter('kdeg_B', 2.0, 1/sec, 'degradation rate'),
    Parameter('kdeg_X', 2.0, 1/sec, 'degradation rate'),
    Parameter('kdeg_Y', 2.0, 1/sec, 'degradation rate'),
])

# species
sc.add([
    Species('A'   , '10+z' , uM            , 100, um**2/sec, 'cytosol'),
    # Species('A'    , 'z'  , uM            , 100, um**2/sec, 'cytosol'),
    #Species('A2'   , '3+z', uM            , 100, um**2/sec, 'cytosol'),
    Species('A2'   , '5+z', uM            , 100, um**2/sec, 'cytosol'),
    Species('A3'   , '7+z'    , uM            , 100, um**2/sec, 'cytosol'),
    Species('B' , 3    , uM            , 100, um**2/sec, 'er_vol'),
    # Species('X' , '100+z'  , molecule/um**2, 10 , um**2/sec, 'pm'),
    # Species('X2' , 40  , molecule/um**2, 10 , um**2/sec, 'pm'),
    # Species('Y', 60  , molecule/um**2, 10 , um**2/sec, 'er_mem'),
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
    # Reaction('B -> 0'       , ['B']     , []    , {'on': 'kdeg_B'}                      , reaction_type='mass_action_forward'    ), # [volume] degradation
    # Reaction('A + X <-> X2' , ['A','X'] , ['X2'], {'on': 'kf_AX_X2', 'off': 'kr_AX_X2'}                                          ), # [volume_to_surface] [surface_to_volume]
    # Reaction('X -> 0'       , ['X']     , []    , {'on': 'kdeg_X'}                      , reaction_type='mass_action_forward'    ), # [surface] degradation
    # Reaction('Y -> 0'       , ['Y']     , []    , {'on': 'kdeg_Y'}                      , reaction_type='mass_action_forward'    ), # [surface] degradation
    # Reaction('A + Y <-> B'  , ['A','Y'] , ['B'] , {'on': 'kf_AY_B'}                     , reaction_type='mass_action_forward'    ), # [volume-surface_to_volume]
    # Reaction('A + B <-> Y'  , ['A','B'] , ['Y'] , {'on': 'kf_AB_Y'}                     , reaction_type='mass_action_forward'    ), # [volume-volume_to_surface]
])

# config
stubs_config = stubs.config.Config()
stubs_config.flags['allow_unused_components'] = True
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
c = model.cc.get_index(0)
r = model.rc.get_index(0)

import cProfile

model._init_1()
model._init_2()
model._init_3()
#cProfile.run("model._init_3()")                 
model._init_4()                 
model._init_5_1_reactions_to_fluxes()
model._init_5_2_create_variational_forms()
#model._init_5_3_create_variational_problems()


#====================

# # aliases
A = model.sc['A']
B = model.sc['B']
# Y = model.sc['Y']
# X = model.sc['X']
# mtot  = model.parent_mesh
# mcyto = model.cc['cytosol'].mesh
# merv  = model.cc['er_vol'].mesh
# merm  = model.cc['er_mem'].mesh
# mpm   = model.cc['pm'].mesh

# ===================
# Nonlinear F==0
# ===================
# Setup
# _extract_args  = d.fem.solving._extract_args
extract_blocks = d.fem.formmanipulations.extract_blocks
# import dolfin.fem.formmanipulations as formmanipulations
# from ufl.algorithms.ad import expand_derivatives
#from dolfin.fem.formmanipulations import derivative
F = sum([f.lhs for f in model.forms])
#F = all_forms_a - all_forms_L
u = model.u['u']

#d.solve(F==0, u)
cProfile.run("d.solve(F==0,u)")                 

#_solve_var_problem(F==0, u)
#eq, u, bcs, J, tol, M, preconditioner, form_compiler_parameters, solver_parameters = _extract_args(F==0, u)
# Extract blocks from the variational formulation
eq = F == 0
eq_lhs_forms = extract_blocks(eq.lhs)

# by compartment
F1 = eq_lhs_forms[0]
F11 = sum([f.lhs for f in model.forms if f.species.name == 'A'])
F2 = eq_lhs_forms[1]
u1 = u._functions[0]
u2 = u._functions[1]
u11 = A._usplit['u']



#if J is None:
# Give the list of jacobian for each eq_lhs
import dolfin.fem.formmanipulations as formmanipulations
from ufl.algorithms.ad import expand_derivatives
import dolfin.cpp as cpp
from dolfin.fem.form import Form
from ufl.form import sub_forms_by_domain


# Jacobian in the form expected by MixedNonlinearVariationalProblem
J = []
for Fi in eq_lhs_forms:
    for uj in u._functions:
        derivative = formmanipulations.derivative(Fi, uj)
        derivative = expand_derivatives(derivative)
        J.append(derivative)

problem = d.MixedNonlinearVariationalProblem(eq_lhs_forms, u._functions, [], J)


# Jacobian in the form given to the cpp object
J_list = list()
print("[problem.py] size J = ", len(J))
for Ji in J:
    if Ji is None:
        J_list.append([cpp.fem.Form(2, 0)])
    elif Ji.empty():
        J_list.append([cpp.fem.Form(2, 0)])
    else:
        Js = []
        for Jsub in sub_forms_by_domain(Ji):
            Js.append(Form(Jsub))
        J_list.append(Js)
    
# cytoJ = Js[0:4]
# pmJ   = Js[4:8]
# ervJ  = Js[8:12] #nonzero
# ermJ  = Js[12:16] #nonzero

# TODO
# look into
# M01 = d.assemble_mixed(J_list[0][1])
# d.PETScNestMatrix (demo_matnest.py)
# it is possible to use petsc4py directly  e.g.
# M = PETSc.Mat().createNest([[M00,M01], [M10,M11]], comm=MPI.COMM_WORLD)
# d.as_backend_type(M00).mat()
#https://fenicsproject.discourse.group/t/custom-newtonsolver-using-the-mixed-dimensional-branch-and-petscnestmatrix/2788/3

# =====================
# solve(F==0, u)
# =====================
# 
#


# =====================
# Dissecting MixedNonlinearVariationalProblem()
# =====================
# dolfin/fem/problem.py
from ufl.form import sub_forms_by_domain
import dolfin.cpp as cpp
from dolfin.fem.form import Form
# aliases
F = eq_lhs_forms
J = Js
u = u._functions

# ====
# Extract and check arguments (u is a list of Function)
u_comps = [u[i]._cpp_object for i in range(len(u))]
# Store input UFL forms and solution Function
F_ufl = eq_lhs_forms
J_ufl = Js
u_ufl = u

assert len(F) == len(u)
assert(len(J) == len(u) * len(u))
# Create list of forms/blocks
F_list = list()
print("[problem.py] size F = ", len(F))
for Fi in F:
    if Fi is None:
        # dolfin/fem/Form.cpp
        # cpp.fem.Form(rank, num_coefficients)
        F_list.append([cpp.fem.Form(1, 0)])
    elif Fi.empty():
        F_list.append([cpp.fem.Form(1, 0)])  # single-elt list
    else:
        Fs = []
        for Fsub in sub_forms_by_domain(Fi):
            if Fsub is None:
                Fs.append(cpp.fem.Form(1, 0))
            elif Fsub.empty():
                Fs.append(cpp.fem.Form(1, 0))
            else:
                Fs.append(Form(Fsub, form_compiler_parameters=form_compiler_parameters))
        F_list.append(Fs)
print("[problem] create list of residual forms OK")

J_list = None
if J is not None:
    J_list = list()
    print("[problem.py] size J = ", len(J))
    for Ji in J:
        if Ji is None:
            J_list.append([cpp.fem.Form(2, 0)])
        elif Ji.empty():
            J_list.append([cpp.fem.Form(2, 0)])
        else:
            Js = []
            for Jsub in sub_forms_by_domain(Ji):
                Js.append(Form(Jsub, form_compiler_parameters=form_compiler_parameters))
            J_list.append(Js)
print("[problem] create list of jacobian forms OK, J_list size = ", len(J_list))

# Initialize C++ base class
cpp.fem.MixedNonlinearVariationalProblem.__init__(self, F_list, u_comps, bcs, J_list)

# # # Create solver and call solve
# solver = d.MixedNonlinearVariationalSolver(problem)
# solver.solve()


# all_forms   = sum([f.form for f in model.forms])
# a_00 = d.extract_blocks(all_forms,0,0)
# a_01 = d.extract_blocks(all_forms,0,1)
# a_10 = d.extract_blocks(all_forms,1,0)
# a_11 = d.extract_blocks(all_forms,1,1)

# d.solve(all_forms == 0, model.u['u']._functions)


# ufl/form.py
# def sub_forms_by_domain(form):
#     "return a list of forms each with an integration domain"
#     if not isinstance(form, form):
#         error("Unable to convert object to a UFL form: %s" % ufl_err_str(form))
#     return [Form(form.integrals_by_domain(domain)) for domain in form.ufl_domains()]