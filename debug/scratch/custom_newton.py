# MWE from
# https://fenicsproject.discourse.group/t/custom-newtonsolver-using-the-mixed-dimensional-branch-and-petscnestmatrix/2788/4
import dolfin as d
import numpy as np

d.parameters["form_compiler"]["quadrature_degree"] = 3

mu = d.Constant(1)
lmbda = d.Constant(1)
h1 = d.Constant((0, 0, 0.1))
h2 = d.Constant((0, 0, -0.1))

mesh = d.UnitCubeMesh(3, 3, 3)


class Top(d.SubDomain):
    def inside(self, x, on_boundary):
        return d.near(x[2], 1.0)


class Bottom(d.SubDomain):
    def inside(self, x, on_boundary):
        return d.near(x[2], 0.0)


top, bot = Top(), Bottom()

marker = d.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
bot.mark(marker, 1)
top.mark(marker, 2)

# Spaces
V = d.VectorFunctionSpace(mesh, "CG", 2)
P = d.FunctionSpace(mesh, "DG", 1)
W = d.MixedFunctionSpace(V, P)

w = d.Function(W)
u, p = w.sub(0), w.sub(1)
(delta_u, delta_us) = d.TestFunctions(W)

dS = d.Measure("ds", domain=W.sub_space(0), subdomain_data=marker)

# Kinematics
I = d.Identity(3)  # Identity tensor
F = I + d.grad(u)  # Deformation gradient
J = d.det(F)
I1C = d.tr(F.T * F / J ** (2.0 / 3.0))

Psi = mu / 2.0 * (I1C - 3.0) - p * (J - 1)  # Strain energy function
Pi = Psi * d.dx + d.inner(h1, u) * dS(1) + d.inner(h2, u) * dS(2)
Gi = [d.derivative(Pi, u), d.derivative(Pi, p)]

dGi = []
for gi in Gi:
    for wi in w.split():
        dGi.append(d.derivative(gi, wi))

# Assemble bilinear form
Ai = []
for dg in dGi:
    Ai.append(d.assemble(dg))
Amat = d.PETScNestMatrix(Ai)
Amat.apply("insert")

# Assemble linear form
bs = []
for g in Gi:
    bs.append(d.assemble(g))

b = d.PETScVector()
Amat.init_vectors(b, bs)

# Set solver
solver_type = "lu"
if solver_type == "krylov":
    solver = d.PETScKrylovSolver()
elif solver_type == "lu":
    solver = d.PETScLUSolver()

Amat.convert_to_aij()
solver.set_operator(Amat)

# Solve
dx0 = np.linalg.solve(Amat.array(), b.get_local())
print("numpy solver:", np.max(dx0))

dx = b.copy()
dx.zero()
solver.solve(dx, b)
print("PETSc solver:", np.max(dx.get_local()))
