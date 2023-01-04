# ===================
# Testing if formmanipulations.derivative(F,u) works on a single Vector Function
# ===================
# mesh = model.parent_mesh.dolfin_mesh
mesh = d.UnitCubeMesh(5, 5, 5)
dx = d.Measure("dx", mesh)

V1 = d.VectorFunctionSpace(mesh, "P", 1, dim=2)
V2 = d.VectorFunctionSpace(mesh, "P", 1, dim=3)
V3 = d.FunctionSpace(mesh, "P", 1)
W = d.MixedFunctionSpace(*[V1, V2, V3])
u = d.Function(W)

v1, v2, v3 = d.TestFunctions(W)
u1, u2, u3 = u.split()

v1, v2, v3 = v.split()

v0 = v[0]
v1 = v[1]
# u0 = u.sub(0); u1 = u.sub(1)
u0, u1 = d.split(u)

F = u0 * v0 * dx + u1 * v1 * dx
G = d.inner(u, v) * dx
i = ufl.indices(1)
H = u[i] * v[i] * dx
all(d.assemble(F) == d.assemble(G))  # True
all(d.assemble(G) == d.assemble(H))  # True

dFdu = dict()
dGdu = dict()

one_vec = d.Expression(("1.0", "1.0"), degree=2, domain=mesh)

dFdu["dolfin.derivative"] = d.derivative(F, u)
dFdu["ufl.derivative"] = ufl.derivative(F, u, v)
# dFdu['ufl.derivative_tuple'] = ufl.derivative(F, (u0,u1))
dFdu["ufl.diff"] = ufl.diff(F, u)


# Argument = ufl.Argument
# def my_derivative(_F, _u):
#     "formmanipulations.derivative()"
#     form_arguments = _F.arguments()
#     number = max([-1] + [arg.number() for arg in form_arguments]) + 1
#     part = _u.part()
#     V = _u.function_space()
#     du = Argument(V, number, part)

#     print(f"V = {V.ufl_element()}")
#     print(f"number = {number}")
#     print(f"part = {part}")
#     #print(f"du = {du}")

#     dFdu = ufl.derivative(_F, _u, du, None)
#     print(f"dFdu = {dFdu}")
#     expanded_derivative = expand_derivatives(dFdu)
#     print(f"dFdu nonzero == {not expanded_derivative.empty()}\n")

#     return expanded_derivative

# my_derivative(_F, _u)
# my_derivative(_F, _u0)
# my_derivative(_F, _u1)

# _dFdu  = expand_derivatives(formmanipulations.derivative(_F, _u))
# _dFdu0 = expand_derivatives(formmanipulations.derivative(_F, _u0))
# _dFdu1 = expand_derivatives(formmanipulations.derivative(_F, _u1))


# print(f"dFdu nonzero == {not _dFdu.empty()}")
# print(f"dFdu{0} nonzero == {not _dFdu0.empty()}")
# print(f"dFdu{1} nonzero == {not _dFdu1.empty()}")
