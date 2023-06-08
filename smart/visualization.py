"""
Utility functions for visualizing SMART meshes using pyvista.
"""
import dolfin
import pyvista
from ufl import vertex
from typing import Tuple
import numpy.typing as npt
import numpy as np
import pathlib
from typing import Optional

# pyvista.global_theme.trame.server_proxy_enabled = True

__all__ = ["create_vtk_structures", "plot"]

_vtk_perm = {
    dolfin.interval: {i: np.arange(i) for i in range(2, 10)},
    vertex: {1: 1},
    dolfin.triangle: {
        1: [0, 1, 2],
        2: [0, 1, 2, 5, 3, 4],
        3: [0, 1, 2, 7, 8, 3, 4, 6, 5, 9],
        4: [0, 1, 2, 9, 10, 11, 3, 4, 5, 8, 7, 6, 12, 13, 14],
    },
    dolfin.tetrahedron: {
        1: [0, 1, 2, 3],
        2: [0, 1, 2, 3, 9, 6, 8, 7, 5, 4],
        3: [0, 1, 2, 3, 14, 15, 8, 9, 13, 12, 10, 11, 6, 7, 4, 5, 18, 16, 17, 19],
    },
}


def create_vtk_structures(
    Vh: dolfin.FunctionSpace,
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.float64]]:
    """Given a (discontinuous) Lagrange space, create pyvista compatible structures based on the
    dof coordinates

    Args:
        Vh: the function space
    Returns:
        Tuple: Mesh topology, cell types and mesh geometry arrays
    """
    family = Vh.ufl_element().family()
    mesh = Vh.mesh()

    num_cells = mesh.num_cells()
    if num_cells == 0:
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros((0, 3), dtype=np.float64),
        )
    d_cell = mesh.ufl_cell()
    # need this extra statement for case of triangles in 3d
    if d_cell._cellname == "triangle":
        d_cell = dolfin.triangle

    if d_cell == vertex:
        cell_type = 1
    elif d_cell == dolfin.interval:
        cell_type = 68
    elif d_cell == dolfin.triangle:
        cell_type = 69
    elif d_cell == dolfin.tetrahedron:
        cell_type = 71
    else:
        raise RuntimeError(f"Unsupported {d_cell=}")
    cell_types = np.full(num_cells, cell_type, dtype=np.int32)
    bs = 1 if Vh.num_sub_spaces() == 0 else Vh.num_sub_spaces()
    Vh = Vh if bs == 1 else Vh.sub(0).collapse()
    num_dofs_per_cell = len(Vh.dofmap().cell_dofs(0))
    topology = np.zeros((num_cells, num_dofs_per_cell + 1), dtype=np.int32)
    topology[:, 0] = num_dofs_per_cell
    dofmap = Vh.dofmap()
    num_vertices_local = dofmap.index_map().size(dofmap.index_map().MapSize.ALL)

    x = np.zeros((num_vertices_local, 3), dtype=np.float64)
    el = Vh.element()
    for i in range(num_cells):
        topology[i, 1:] = dofmap.cell_dofs(i)
        cell = dolfin.Cell(mesh, i)
        cell_dof_coords = el.tabulate_dof_coordinates(cell)
        for j, d in enumerate(dofmap.cell_dofs(i)):
            x[d, : cell_dof_coords.shape[1]] = cell_dof_coords[j]

    degree = Vh.ufl_element().degree()
    try:
        perm = _vtk_perm[d_cell][degree]
    except KeyError:
        raise RuntimeError(f"Unsupported plotting of space {family} of {degree=} on {d_cell}")
    topology[:, 1:] = topology[:, 1:][:, perm]
    return topology, cell_types, x


def plot(
    uh: dolfin.Function,
    filename: Optional[pathlib.Path] = None,
    show_edges: bool = True,
    glyph_factor: float = 1,
    off_screen: bool = True,
    view_xy: bool = False,
):
    """
    Plot a (discontinuous) Lagrange function with Pyvista

    Args:
        uh: The function
        filename: If set, writes the plot to file instead of displaying it interactively
        show_edges: Show mesh edges if ``True``
        glyph_factor: Scaling of glyphs if input function is a function from a
            ``dolfin.VectorFunctionSpace``.
        off_screen: If ``True`` generate plots with virtual frame buffer using ``xvfb``.
        view_xy: If ``True``, view xy plane
    """
    Vh = uh.function_space()

    dofmap = Vh.dofmap()
    num_vertices_local = dofmap.index_map().size(dofmap.index_map().MapSize.ALL)
    bs = 1 if Vh.num_sub_spaces() == 0 else Vh.num_sub_spaces()

    u_vec = uh.vector().get_local(np.arange(num_vertices_local))
    if bs > 1:
        u_out = np.zeros((len(u_vec) // bs, 3), dtype=np.float64)
        u_out[:, :bs] = u_vec.reshape(len(u_vec) // bs, bs)
    else:
        u_out = u_vec
    topology, cell_types, x = create_vtk_structures(Vh)

    grid = pyvista.UnstructuredGrid(topology, cell_types, x)
    grid[uh.name()] = u_out

    if bs > 1:
        grid.set_active_scalars(None)

    pyvista.OFF_SCREEN = off_screen
    pyvista.start_xvfb()

    plotter = pyvista.Plotter()

    # Workaround for pyvista issue for higherorder grids
    degree = Vh.ufl_element().degree()
    if degree > 1 and show_edges:
        first_order_el = Vh.ufl_element().reconstruct(degree=1)
        P1 = dolfin.FunctionSpace(Vh.mesh(), first_order_el)
        t1, c1, x1 = create_vtk_structures(P1)
        p1_grid = pyvista.UnstructuredGrid(t1, c1, x1)
        if bs > 1:
            plotter.add_mesh(p1_grid, show_edges=show_edges)
        else:
            plotter.add_mesh(grid)
            plotter.add_mesh(p1_grid, style="wireframe")
    else:
        if uh.geometric_dimension() == 2 or view_xy:
            plotter.add_mesh(grid, show_edges=show_edges)
        else:
            crinkled = grid.clip(normal=(1, 0, 0), crinkle=True)
            plotter.add_mesh(crinkled, show_edges=show_edges)

    if uh.geometric_dimension() == 2 or view_xy:
        plotter.view_xy()

    if bs > 1:
        glyphs = grid.glyph(orient=uh.name(), factor=glyph_factor)
        plotter.add_mesh(glyphs)
    print(filename)
    if filename is None:
        plotter.show()
    else:
        plotter.screenshot(filename)


def plot_dolfin_mesh(
    msh: dolfin.mesh,
    mf_cell: dolfin.MeshFunction,
    mf_facet: dolfin.MeshFunction = None,
    outer_marker: int = 10,
    filename: Optional[pathlib.Path] = None,
    show_edges: bool = True,
    off_screen: bool = True,
    view_xy: bool = False,
):
    """
    Construct P1 function space on current mesh,
    convert function space to vtk structures,
    and then plot mesh markers in mf (a dolfin MeshFunction)

    Args:
        msh: dolfin mesh object
        mf_cell: marker values for cells in domain, given as a dolfin MeshFunction
        mf_facet (optional): marker values for facets at domain boundary,
            given as a dolfin MeshFunction defined over all facets in the mesh
        outer_marker (optional): value marking the boundary facets in mf_facet.
            Number of nodes with this marker
            should match the number of nodes in dolfin.BoundaryMesh(msh, "exterior")
        filename: If set, writes the plot to file instead of displaying it interactively
        show_edges: Show mesh edges if ``True``
        off_screen: If ``True`` generate plots with virtual frame buffer using ``xvfb``.
        view_xy: If ``True``, view xy plane
    """
    Vh = dolfin.FunctionSpace(msh, "P", 1)
    u_out = mf_cell.array()
    topology, cell_types, x = create_vtk_structures(Vh)
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)
    grid["mf"] = u_out

    facets_loaded = False
    if mf_facet is not None:
        msh_facet = dolfin.BoundaryMesh(msh, "exterior")
        Vh_facet = dolfin.FunctionSpace(msh_facet, "P", 1)
        u_facet = mf_facet.array()[
            mf_facet.array() == outer_marker
        ]  # outer_marker labels outer surface
        topology, cell_types, x = create_vtk_structures(Vh_facet)
        grid_facet = pyvista.UnstructuredGrid(topology, cell_types, x)
        try:
            grid_facet["mf"] = u_facet
            facets_loaded = True
        except ValueError:
            facets_loaded = False

    pyvista.OFF_SCREEN = off_screen
    pyvista.start_xvfb()

    plotter = pyvista.Plotter()

    # Workaround for pyvista issue for higherorder grids
    degree = Vh.ufl_element().degree()
    if degree > 1 and show_edges:
        first_order_el = Vh.ufl_element().reconstruct(degree=1)
        P1 = dolfin.FunctionSpace(Vh.mesh(), first_order_el)
        t1, c1, x1 = create_vtk_structures(P1)
        p1_grid = pyvista.UnstructuredGrid(t1, c1, x1)
        plotter.add_mesh(grid)
        plotter.add_mesh(p1_grid, style="wireframe")
    else:
        if msh.topology().dim() == 3:
            crinkled = grid.clip(normal=(1, 0, 0), crinkle=True)
            plotter.add_mesh(crinkled, show_edges=show_edges)
            if facets_loaded:
                crinkled_facet = grid_facet.clip(normal=(1, 0, 0), crinkle=True)
                plotter.add_mesh(crinkled_facet, show_edges=show_edges)
        elif msh.topology().dim() == 2:
            plotter.add_mesh(grid, show_edges=show_edges)
            if facets_loaded:
                plotter.add_mesh(grid_facet, show_edges=show_edges)

    if msh.geometric_dimension() == 2 or view_xy:
        plotter.view_xy()

    print(filename)
    if filename is None:
        plotter.show()
    else:
        plotter.screenshot(filename)
