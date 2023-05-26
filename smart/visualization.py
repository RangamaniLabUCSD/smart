"""
Utility functions for visualizing SMART meshes using pyvista.
"""

import pathlib

import pyvista
import dolfin as d
import meshio


def plot_dolfin_mesh(msh: d.mesh, mf: d.Function):
    """
    Save, then read in dolfin mesh to pyvista
    Plot mesh with markers of indicated dimensionality

    """
    dim = mf.dim()
    tmp_folder = pathlib.Path(f"tmp_mesh_d{dim}")
    tmp_folder.mkdir(exist_ok=True)
    mesh_file_path = tmp_folder / "mesh_function.xdmf"
    mesh_file = d.XDMFFile(str(mesh_file_path))
    mesh_file.parameters["flush_output"] = True
    mesh_file.write(msh)
    mesh_in = meshio.read(str(mesh_file_path))
    mesh_file_vtk = tmp_folder / "mesh_function.vtk"
    meshio.write(str(mesh_file_vtk), mesh_in)
    mesh_load = pyvista.read(str(mesh_file_vtk))
    mesh_load.cell_data["Marker"] = mf.array()
    mesh_load.set_active_scalars("Marker")
    plotter = pyvista.Plotter()
    plotter.add_mesh(mesh_load, show_edges=True, opacity=0.5)
    if dim == 2:
        plotter.view_xy()
    plotter.show()
    # remove tmp files and dir
    mesh_file_path.unlink(missing_ok=False)
    mesh_file_path.with_suffix(".h5").unlink(missing_ok=False)
    mesh_file_vtk.unlink(missing_ok=False)
    tmp_folder.rmdir()


def plot_dolfin_function(msh: d.mesh, u: d.Function):
    """
    Save, then read in dolfin mesh to pyvista
    Plot mesh with function values at each vertex

    """
    dim = u.geometric_dimension()
    tmp_folder = pathlib.Path(f"tmp_mesh_d{dim}")
    tmp_folder.mkdir(exist_ok=True)
    mesh_file_path = tmp_folder / "mesh_function.xdmf"
    mesh_file = d.XDMFFile(str(mesh_file_path))
    mesh_file.parameters["flush_output"] = True
    mesh_file.write(msh)
    mesh_in = meshio.read(str(mesh_file_path))
    mesh_file_vtk = tmp_folder / "mesh_function.vtk"
    meshio.write(str(mesh_file_vtk), mesh_in)
    mesh_load = pyvista.read(str(mesh_file_vtk))
    mesh_load.point_data["u"] = u.compute_vertex_values()
    mesh_load.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(mesh_load, show_edges=True)
    if dim == 2:
        plotter.view_xy()
    plotter.show()
    # remove tmp files and dir
    mesh_file_path.unlink(missing_ok=False)
    mesh_file_path.with_suffix(".h5").unlink(missing_ok=False)
    mesh_file_vtk.unlink(missing_ok=False)
    tmp_folder.rmdir()
