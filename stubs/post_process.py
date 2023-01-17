
__all__ = ["get_tvec", "set_u", "integrate"]

try:
    import h5py
    __all__ += ["get_data", "get_mesh"]
except ModuleNotFoundError:
    print("h5py is not installed, functions `get_data` and `get_mesh` not exposed")

import dolfin as d
import numpy as np
COMM = d.MPI.comm_world


def get_tvec(dirpath):
    with open(dirpath + 'tvec.txt', 'r') as file:
        tvec = [float(t) for t in file.readlines()]
    return tvec


def get_data(h5name, uname, tidx):
    with h5py.File(h5name, 'r') as h5file:
        data = h5file[f'/{uname}/{tidx}'].value
    return data.reshape(-1)


def get_mesh(h5name):
    with h5py.File(h5name, 'a') as h5file:
        hmesh = h5file["Mesh/"]
        hmesh["topology"].attrs['celltype'] = np.bytes_('tetrahedron')
        if "coordinates" not in hmesh.keys():
            hmesh["coordinates"] = hmesh["geometry"]
        if "cell_indices" not in hmesh.keys():
            hmesh["cell_indices"] = np.arange(hmesh["topology"].shape[0])

    dmesh = d.Mesh(COMM)
    hdf5 = d.HDF5File(dmesh.mpi_comm(), h5name, 'r')
    hdf5.read(dmesh, 'Mesh', False)
    hdf5.close()
    V = d.FunctionSpace(dmesh, 'CG', 1)
    return dmesh, V


def set_u(V, u, u_):
    dofs = d.vertex_to_dof_map(V)
    u.vector()[dofs] = u_
    u.vector().apply('insert')


def integrate(u):
    return d.assemble(u*d.dx(domain=u.function_space().mesh()))
