from dolfin import *

##### ---------- Parameters ----------- #####
alpha = 1.0
# mu : Dynamic viscosity
# water : 1e-3 kg.m-1.s-1
mu = 1e-3
# rho : fluid density
# water : 1000 kg.m-3
rho = 1e3
# p0_in/p0_out : Imposed pressures at inlet/outlet 
p0_in = 8
p0_out = 10
f = Constant(("1e-3","0.0"))
##### -------------------------------- #####

##### ---------- Meshes - Function Spaces ----------- #####
n1 = 80
n2 = 20
mesh = RectangleMesh(Point(0.0, 0.0), Point(10.0, 5.0), n1, n2)
physical_facets = MeshFunction("size_t", mesh, 1, 0)

CompiledSubDomain('near(x[0], 0)').mark(physical_facets, 101)
CompiledSubDomain('near(x[0], 10.0)').mark(physical_facets, 102)
CompiledSubDomain('near(x[1]*(x[1]-5.0), 0)').mark(physical_facets, 100)

# Subdomains markers : 100 = wall, 101 = inlet, 102 = outlet
## 1D meshes
inlet_mesh = MeshView.create(physical_facets, 101) ## Gamma1
outlet_mesh = MeshView.create(physical_facets, 102) ## Gamma2

# P2-P1 for Stokes
V = VectorFunctionSpace(mesh, "CG", 2)
M = FunctionSpace(mesh, "CG", 1)
# P2 for both Lagrange multipliers (inlet / outlet)
LM_in  = FunctionSpace(inlet_mesh, "CG", 2)
LM_out  = FunctionSpace(outlet_mesh, "CG", 2)
W = MixedFunctionSpace(V, M, LM_in, LM_out)

# Trial (u,p,lambda_in,lambda_out)
(u, p, li, lo) = TrialFunctions(W)
# Test (v,q,eta_in,eta_out)
(v, q, ei, eo) = TestFunctions(W)

dV = Measure("dx", domain=W.sub_space(0).mesh()) # Integration on the blood vessel - velocity
dL_in = Measure("dx", domain=W.sub_space(2).mesh()) # Integration on the inlet boundary - LM_in
dL_out = Measure("dx", domain=W.sub_space(3).mesh()) # Integration on the outlet boundary - LM_out

# Boundaries as exterior facets from 2D mesh
dV_boundary = Measure("ds", domain=W.sub_space(0).mesh(), subdomain_data=physical_facets)
# Boundaries for LMs
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
# Mark boundary of LM_in
boundaries_LM_in = MeshFunction("size_t", inlet_mesh, inlet_mesh.topology().dim()-1, 0)
DirichletBoundary().mark(boundaries_LM_in, 1)
# Mark boundary of LM_out
boundaries_LM_out = MeshFunction("size_t", outlet_mesh, outlet_mesh.topology().dim()-1, 0)
DirichletBoundary().mark(boundaries_LM_out, 1)

##### ----------------------------------------------- #####

##### -------------- Variational form --------------- #####

n = FacetNormal(mesh)
t_lm = Expression(("0.0","-1.0"), degree=1)

defu = sym(grad(u))
defv = sym(grad(v))

# mu = constant, defu = grad u
# dV - volume (2d)
# dL_in/dL_out - dx (1d) inlet/outlet boundary

a0 = 2*mu*inner(defu,defv)*dV - div(v)*p*dV - div(u)*q*dV 
cvl = li*inner(t_lm,v)*dL_in + lo*inner(t_lm,v)*dL_out
cue = ei*inner(t_lm,u)*dL_in + eo*inner(t_lm,u)*dL_out

foo = Constant(0)*inner(li, ei)*dL_in + Constant(0)*inner(lo, eo)*dL_out
## Bilinear form
a = a0 - cvl - cue + foo
## Linear form
L = rho*inner(f,v)*dV - p0_in*inner(v,n)*dV_boundary(101) - p0_out*inner(v,n)*dV_boundary(102)

##### ----------------------------------------------- #####

##### ----------- Boundary conditions --------------- #####

# Boundary conditions on the wall
zero = Constant(("0.0","0.0"))
bc_wall = DirichletBC(W.sub_space(0), zero, physical_facets, 100)
# Boundary conditions for LMs
bc_lm_in = DirichletBC(W.sub_space(2), Constant(0), boundaries_LM_in, 1)
bc_lm_out = DirichletBC(W.sub_space(3), Constant(0), boundaries_LM_out, 1)
bcs = [bc_wall, bc_lm_in, bc_lm_out]

##### ------------------- Solve --------------------- #####

sol = Function(W)
solve(a == L, sol, bcs, solver_parameters={"linear_solver":"direct"})

out_u = File("Stokes-TractionBCs-v.pvd")
out_p = File("Stokes-TractionBCs-p.pvd")
out_li = File("Stokes-TractionBCs-li.pvd")
out_lo = File("Stokes-TractionBCs-lo.pvd")
out_u << sol.sub(0)
out_p << sol.sub(1)
out_li << sol.sub(2)
out_lo << sol.sub(3)