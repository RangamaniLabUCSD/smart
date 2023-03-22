__all__ = ["get_tvec", "set_u", "integrate"]

from .deprecation import deprecated
import dolfin as d


@deprecated
def get_tvec(dirpath):
    with open(dirpath + "tvec.txt", "r") as file:
        tvec = [float(t) for t in file.readlines()]
    return tvec


@deprecated
def set_u(V, u, u_):
    dofs = d.vertex_to_dof_map(V)
    u.vector()[dofs] = u_
    u.vector().apply("insert")


@deprecated
def integrate(u):
    return d.assemble(u * d.dx(domain=u.function_space().mesh()))
