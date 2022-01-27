"""Scratch code for whatever I'm trying to figure out at the moment"""

import stubs
import stubs.common as common
import dolfin as d
import ufl
import numpy as np
import pint
unit = stubs.unit # unit registry

#====================
# aliases - unit registry
# ===================
from stubs.model_assembly import Parameter, Species, Compartment, Reaction
uM       = unit.uM
meter    = unit.m
um       = unit.um
molecule = unit.molecule
sec      = unit.s

import stubs_model 
model = stubs_model.make_model(refined_mesh=True)

#====================
# init model
# ===================
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
# aliases
# ===================
p = model.pc.get_index(0)
s = model.sc.get_index(0)
c = model.cc.get_index(0)
r = model.rc.get_index(0)
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
# Imports
# _extract_args  = d.fem.solving._extract_args
from dolfin.fem.formmanipulations import derivative, extract_blocks
import dolfin.cpp as cpp
from dolfin.fem.form import Form
from ufl.algorithms.ad import expand_derivatives
from ufl.form import sub_forms_by_domain


# Aliases 
dfdu = lambda F,u: expand_derivatives(d.derivative(F,u))
F = sum([f.lhs for f in model.forms]) # single form
u = model.u['u']
Fblock = extract_blocks(F)

Flist, Jlist = model.get_block_system()

d.solve(F==0, u)

# TODO
# look into
# M01 = d.assemble_mixed(Jlist[0][1])
# d.PETScNestMatrix (demo_matnest.py)
# it is possible to use petsc4py directly  e.g.
# M = PETSc.Mat().createNest([[M00,M01], [M10,M11]], comm=MPI.COMM_WORLD)
# d.as_backend_type(M00).mat()
#https://fenicsproject.discourse.group/t/custom-newtonsolver-using-the-mixed-dimensional-branch-and-petscnestmatrix/2788/3


"""
Solve with high-level call, d.solve(F==0, u)
When F is multi-dimensional + non-linear d.solve() roughly executes the following:

Fsum = sum([f.lhs for f in model.forms]) # single form F0+F1+...+Fn

* d.solve(Fsum==0, u)                                                       [fem/solving.py]
    * _solve_varproblem()                                                   [fem/solving.py]
        eq, ... = _extract_args()
        F = extract_blocks(eq.lhs) # tuple of forms (F0, F1, ..., Fn)       [fem/formmanipulations -> ufl/algorithms/formsplitter]
        for Fi in F:
            for uj in u._functions:
                Js.append(expand_derivatives(formmanipulations.derivative(Fi, uj)))
                # [J00, J01, J02, etc...]
        problem = MixedNonlinearVariationalProblem(F, u._functions, bcs, Js)
        solver  = MixedNonlinearVariationalSolver(problem)
        solver.solve()

* MixedNonlinearVariationalProblem(F, u._functions, bcs, Js)     [fem/problem.py] 
    u_comps = [u[i]._cpp_object for i in range(len(u))] 

    # if len(F)!= len(u) -> Fill empty blocks of F with None

    # Check that len(J)==len(u)**2 and len(F)==len(u)

    # use F to create Flist. Separate forms by domain:
    # Flist[i] is a list of Forms separated by domain. E.g. if F1 consists of integrals on \Omega_1, \Omega_2, and \Omega_3
    # then Flist[i] is a list with 3 forms
    If Fi is None -> Flist[i] = cpp.fem.Form(1,0) 
    else -> Flist[i] = [Fi[domain=0], Fi[domain=1], ...]

    # Do the same for J -> Jlist

    cpp.fem.MixedNonlinearVariationalProblem.__init__(self, Flist, u_comps, bcs, Jlist)
    
    

========
Notes:
========
# on extract_blocks(F)
F  = sum([f.lhs for f in model.forms]) # single form
Fb = extract_blocks(F) # tuple of forms
Fb0 = Fb[0]

F0 = sum([f.lhs for f in model.forms if f.compartment.name=='cytosol'])
F0.equals(Fb0) -> False

I0 = F0.integrals()[0].integrand()
Ib0 = Fb0.integrals()[0].integrand()
I0.ufl_operands[0] == Ib0.ufl_operands[0] -> False (ufl.Indexed(Argument))) vs ufl.Indexed(ListTensor(ufl.Indexed(Argument)))
I0.ufl_operands[1] == Ib0.ufl_operands[1] -> True
I0.ufl_operands[0] == Ib0.ufl_operands[0](1) -> True


# on d.functionspace
V.__repr__() shows the UFL coordinate element (finite element over coordinate vector field) and finite element of the function space.
We can access individually with:
V.ufl_domain().ufl_coordinate_element()
V.ufl_element()

# on assembler
d.fem.assembling.assemble_mixed(form, tensor)
assembler = cpp.fem.MixedAssembler()


fem.assemble.cpp/assemble_mixed(GenericTensor& A, const Form& a, bool add)
  MixedAssembler assembler;
  assembler.add_values = add;
  assembler.assemble(A, a);
"""


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



# =====================
# solve(F==0, u)
# =====================
# 
#


# =====================
# Dissecting MixedNonlinearVariationalProblem()
# =====================
flag = False
if flag == True:
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
