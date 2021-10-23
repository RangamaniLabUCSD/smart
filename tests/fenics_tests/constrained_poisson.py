from dolfin import *
# parameters
n = 16
EPS = DOLFIN_EPS
# Generate the meshes
mesh = UnitSquareMesh(n, n)

# MF is defined on interval elements. Initialize meshfunction to 0.
marker = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
for f in facets(mesh):
    # is the x-coordinate of the midpoint of a facet close to 0.5?
    marker[f] = 0.5 - EPS < f.midpoint().x() < 0.5 + EPS

submesh = MeshView.create(marker, 1)

# Initialize function spaces and basis functions
V = FunctionSpace(mesh, "CG", 1)
LM = FunctionSpace(submesh, "CG", 1)
W = MixedFunctionSpace(V,LM)
(u,l) = TrialFunctions(W)
(v,e) = TestFunctions(W)
# Dirichlet boundary condition(x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

bc = DirichletBC(V, Constant(0.0), boundary)

# Variational formulation
dV = Measure("dx", domain=W.sub_space(0).mesh())
dL = Measure("dx", domain=W.sub_space(1).mesh())
a = inner(grad(u),grad(v))*dV + v*l*dL + u*e*dL
L = Constant(2)*v*dV + Constant(0.25)*e*dL

# Solve the problem
sol = Function(W)
solve(a == L, sol, bc)