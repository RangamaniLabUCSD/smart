from dolfin import *

#if has_linear_algebra_backend("Epetra"):
    #parameters["linear_algebra_backend"] = "Epetra"

# Sub domain for Dirichlet boundary condition
class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 1.0) < DOLFIN_EPS and on_boundary

# Create mesh and define function space
n = 32
mesh = UnitSquareMesh(n, n)
V = FunctionSpace(mesh, "CG", 1)

# Define boundary condition
g = Constant(1.0)
bc = DirichletBC(V, g, DirichletBoundary())

# Define variational problem
u = Function(V)
v = TestFunction(V)
f = Expression("x[0]*sin(x[1])", degree=1)
F = inner((1 + u**2)*grad(u), grad(v))*dx - f*v*dx

# Compute solution
solve(F == 0, u, bc, solver_parameters={"newton_solver":
                                        {"relative_tolerance": 1e-6}})


print(u.vector().get_local().max())
