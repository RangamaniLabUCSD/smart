#tensors test
import dolfin as d
mesh = d.UnitIntervalMesh(4)

A = d.PETScMatrix()

V = d.FunctionSpace(mesh, "CG", 1)
u = d.interpolate(d.Constant(2), V)
ut = d.TrialFunction(V)
v = d.TestFunction(V)
a = d.inner(d.grad(ut), d.grad(v))*d.dx
d.assemble(a, tensor=A)#, finalize_tensor=False)