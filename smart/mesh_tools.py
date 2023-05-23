from typing import Tuple
import pathlib
import numpy as np
import dolfin as d
from mpi4py import MPI

__all__ = [
    "facet_topology",
    "cube_condition",
    "DemoCuboidsMesh",
    "DemoSpheresMesh",
    "DemoEllipsoidsMesh",
    "DemoEllipseMesh",
    "write_mesh",
]


def facet_topology(f: d.Facet, mf3: d.MeshFunction):
    """Given a facet and cell mesh function,
    return the topology of the face"""
    # cells adjacent face
    localCells = [mf3.array()[c.index()] for c in d.cells(f)]
    if len(localCells) == 1:
        topology = "boundary"  # boundary facet
    elif len(localCells) == 2 and localCells[0] == localCells[1]:
        topology = "internal"  # internal facet
    elif len(localCells) == 2:
        topology = "interface"  # interface facet
    else:
        raise Exception("Facet has more than two cells")
    return (topology, localCells)


def cube_condition(cell, xmin=0.3, xmax=0.7):
    return (
        (xmin - d.DOLFIN_EPS < cell.midpoint().x() < xmax + d.DOLFIN_EPS)
        and (xmin - d.DOLFIN_EPS < cell.midpoint().y() < xmax + d.DOLFIN_EPS)
        and (xmin - d.DOLFIN_EPS < cell.midpoint().z() < xmax + d.DOLFIN_EPS)
    )


def DemoCuboidsMesh(N=16, condition=cube_condition):
    """
    Creates a mesh for use in examples that contains
    two distinct cuboid subvolumes with a shared interface surface.
    Cell markers:
    1 - Default subvolume
    2 - Subvolume specified by condition function

    Facet markers:
    12 - Interface between subvolumes
    10 - Boundary of subvolume 1
    20 - Boundary of subvolume 2
    0  - Interior facets
    """
    # Create a mesh
    mesh = d.UnitCubeMesh(N, N, N)
    # Initialize mesh functions
    mf3 = d.MeshFunction("size_t", mesh, 3, 0)
    mf2 = d.MeshFunction("size_t", mesh, 2, 0)

    # Mark all cells that satisfy condition as 3, else 1
    for c in d.cells(mesh):
        mf3[c] = 2 if condition(c) else 1

    # Mark facets
    for f in d.faces(mesh):
        topology, cellIndices = facet_topology(f, mf3)
        if topology == "interface":
            mf2[f] = 12
        elif topology == "boundary":
            mf2[f] = int(cellIndices[0] * 10)
        else:
            mf2[f] = 0
    return (mesh, mf2, mf3)


def DemoSpheresMesh(
    outerRad: float = 0.5,
    innerRad: float = 0.25,
    hEdge: float = 0,
    hInnerEdge: float = 0,
    interface_marker: int = 12,
    outer_marker: int = 10,
    inner_vol_tag: int = 2,
    outer_vol_tag: int = 1,
    comm: MPI.Comm = d.MPI.comm_world,
) -> Tuple[d.Mesh, d.MeshFunction, d.MeshFunction]:
    """
    Calls DemoEllipsoidsMesh() to make spherical mesh
    """
    dmesh, mf2, mf3 = DemoEllipsoidsMesh(
        (outerRad, outerRad, outerRad),
        (innerRad, innerRad, innerRad),
        hEdge,
        hInnerEdge,
        interface_marker,
        outer_marker,
        inner_vol_tag,
        outer_vol_tag,
        comm,
    )
    return (dmesh, mf2, mf3)


def DemoEllipsoidsMesh(
    outerRad: Tuple[float, float, float],
    innerRad: Tuple[float, float, float],
    hEdge: float = 0,
    hInnerEdge: float = 0,
    interface_marker: int = 12,
    outer_marker: int = 10,
    inner_vol_tag: int = 2,
    outer_vol_tag: int = 1,
    comm: MPI.Comm = d.MPI.comm_world,
) -> Tuple[d.Mesh, d.MeshFunction, d.MeshFunction]:
    """
    Creates a mesh for use in examples that contains
    two distinct ellipsoid subvolumes with a shared interface
    surface. If the radius of the inner ellipsoid is 0, mesh a
    single ellipsoid.

    Args:
        outerRad: The radius of the outer ellipsoid
        innerRad: The radius of the inner ellipsoid
        hEdge: maximum mesh size at the outer edge
        hInnerEdge: maximum mesh size at the edge
        of the inner ellipsoid interface_marker: The
        value to mark facets on the interface with
        outer_marker: The value to mark facets on the outer ellipsoid with
        inner_vol_tag: The value to mark the inner spherical volume with
        outer_vol_tag: The value to mark the outer spherical volume with
        comm: MPI communicator to create the mesh with
    Returns:
        A triplet (mesh, facet_marker, cell_marker)
    """
    import gmsh

    if np.any(np.isclose(outerRad, 0)):
        ValueError("One of the outer radii is equal to zero")
    if np.isclose(hEdge, 0):
        hEdge = 0.1 * max(outerRad)
    if np.isclose(hInnerEdge, 0):
        hInnerEdge = 0.2 * max(outerRad) if np.any(np.isclose(innerRad, 0)) else 0.2 * max(innerRad)
    if innerRad[0] > outerRad[0] or innerRad[1] > outerRad[1] or innerRad[2] > outerRad[2]:
        ValueError("Inner ellipsoid does not fit inside outer ellipsoid")
    # Create the two ellipsoid mesh using gmsh
    gmsh.initialize()
    gmsh.model.add("twoellipsoids")
    # first add ellipsoid 1 of radius outerRad and center (0,0,0)
    outer_ellipsoid = gmsh.model.occ.addSphere(0, 0, 0, 1.0)
    gmsh.model.occ.dilate([(3, outer_ellipsoid)], 0, 0, 0, outerRad[0], outerRad[1], outerRad[2])
    if np.any(np.isclose(innerRad, 0)):
        # Use outer_ellipsoid only
        gmsh.model.occ.synchronize()
        gmsh.model.add_physical_group(3, [outer_ellipsoid], tag=outer_vol_tag)
        facets = gmsh.model.getBoundary([(3, outer_ellipsoid)])
        assert len(facets) == 1
        gmsh.model.add_physical_group(2, [facets[0][1]], tag=outer_marker)
    else:
        # Add inner_ellipsoid (radius innerRad, center (0,0,0))
        inner_ellipsoid = gmsh.model.occ.addSphere(0, 0, 0, 1.0)
        gmsh.model.occ.dilate(
            [(3, inner_ellipsoid)], 0, 0, 0, innerRad[0], innerRad[1], innerRad[2]
        )
        # Create interface between ellipsoids
        two_ellipsoids, (outer_ellipsoid_map, inner_ellipsoid_map) = gmsh.model.occ.fragment(
            [(3, outer_ellipsoid)], [(3, inner_ellipsoid)]
        )
        gmsh.model.occ.synchronize()

        # Get the outer boundary
        outer_shell = gmsh.model.getBoundary(two_ellipsoids, oriented=False)
        assert len(outer_shell) == 1
        # Get the inner boundary
        inner_shell = gmsh.model.getBoundary(inner_ellipsoid_map, oriented=False)
        assert len(inner_shell) == 1
        # Add physical markers for facets
        gmsh.model.add_physical_group(outer_shell[0][0], [outer_shell[0][1]], tag=outer_marker)
        gmsh.model.add_physical_group(inner_shell[0][0], [inner_shell[0][1]], tag=interface_marker)

        # Physical markers for
        all_volumes = [tag[1] for tag in outer_ellipsoid_map]
        inner_volume = [tag[1] for tag in inner_ellipsoid_map]
        outer_volume = []
        for vol in all_volumes:
            if vol not in inner_volume:
                outer_volume.append(vol)
        gmsh.model.add_physical_group(3, outer_volume, tag=outer_vol_tag)
        gmsh.model.add_physical_group(3, inner_volume, tag=inner_vol_tag)

    def meshSizeCallback(dim, tag, x, y, z, lc):
        # mesh length is hEdge at the PM (defaults to 0.1*outerRad,
        # or set when calling function) and hInnerEdge at the ERM
        # (defaults to 0.2*innerRad, or set when calling function)
        # between these, the value is interpolated based on R,
        # and inside the value is interpolated between hInnerEdge and 0.2*innerEdge
        # if innerRad=0, then the mesh length is interpolated between
        # hEdge at the PM and 0.2*outerRad in the center
        # for one ellipsoid (innerRad = 0), if hEdge > 0.2*outerRad,
        # then lc = 0.2*outerRad in the whole volume
        # for two ellipsoids, if hEdge or hInnerEdge > 0.2*innerRad,
        # they are set to lc = 0.2*innerRad
        R_rel_outer = np.sqrt(
            (x / outerRad[0]) ** 2 + (y / outerRad[1]) ** 2 + (z / outerRad[2]) ** 2
        )
        if np.any(np.isclose(innerRad, 0)):
            lc3 = 0.2 * max(outerRad)
            innerRad_scale = 0
            in_outer = True
        else:
            R_rel_inner = np.sqrt(
                (x / innerRad[0]) ** 2 + (y / innerRad[1]) ** 2 + (z / innerRad[2]) ** 2
            )
            lc3 = 0.2 * max(innerRad)
            innerRad_scale = np.mean(
                [innerRad[0] / outerRad[0], innerRad[1] / outerRad[1], innerRad[2] / outerRad[2]]
            )
            in_outer = R_rel_inner > 1
        lc1 = hEdge
        lc2 = hInnerEdge
        if in_outer:
            lcTest = lc1 + (lc2 - lc1) * (1 - R_rel_outer) / (1 - innerRad_scale)
        else:
            lcTest = lc2 + (lc3 - lc2) * (1 - R_rel_inner)
        return min(lc3, lcTest)

    gmsh.model.mesh.setSizeCallback(meshSizeCallback)
    # set off the other options for mesh size determination
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    # this changes the algorithm from Frontal-Delaunay to Delaunay,
    # which may provide better results when there are larger gradients in mesh size
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    gmsh.model.mesh.generate(3)
    rank = MPI.COMM_WORLD.rank
    tmp_folder = pathlib.Path(f"tmp_ellipsoid_{outerRad}_{innerRad}_{rank}")
    tmp_folder.mkdir(exist_ok=True)
    gmsh_file = tmp_folder / "ellipsoids.msh"
    gmsh.write(str(gmsh_file))
    gmsh.finalize()

    # return dolfin mesh of max dimension (parent mesh) and marker functions mf2 and mf3
    dmesh, mf2, mf3 = gmsh_to_dolfin(str(gmsh_file), tmp_folder, 3, comm)
    # remove tmp mesh and tmp folder
    gmsh_file.unlink(missing_ok=False)
    tmp_folder.rmdir()
    # return dolfin mesh, mf2 (2d tags) and mf3 (3d tags)
    return (dmesh, mf2, mf3)


def DemoEllipseMesh(
    xrad: float = 3.0,
    yrad: float = 1.0,
    h_ellipse: float = 0.1,
    inside_tag: int = 1,
    edge_tag: int = 3,
    comm: MPI.Comm = d.MPI.comm_world,
) -> Tuple[d.Mesh, d.MeshFunction, d.MeshFunction]:
    """
    Creates a mesh for an ellipse surface
    Args:
        xrad: radius assoc with major axis
        yrad: radius assoc with minor axis
        h_ellipse: mesh resolution
        inside_tag: mesh marker value for triangles in the ellipse
        edge_tag: mesh marker value for edge 1D elements
        comm: MPI communicator to create the mesh with
    Returns:
        A triplet (mesh, facet_marker (mf1), cell_marker(mf2))
    """
    import gmsh

    assert not np.isclose(xrad, 0) and not np.isclose(yrad, 0)
    if np.isclose(h_ellipse, 0):
        h_ellipse = 0.1 * min((xrad, yrad))
    # Create the ellipse mesh
    gmsh.initialize()
    gmsh.model.add("ellipse")
    # first add ellipse curve
    ellipse = gmsh.model.occ.addDisk(0, 0, 0, xrad, yrad)
    gmsh.model.occ.synchronize()
    gmsh.model.add_physical_group(2, [ellipse], tag=inside_tag)
    facets = gmsh.model.getBoundary([(2, ellipse)])
    assert len(facets) == 1
    gmsh.model.add_physical_group(1, [facets[0][1]], tag=edge_tag)

    def meshSizeCallback(dim, tag, x, y, z, lc):
        return h_ellipse

    gmsh.model.mesh.setSizeCallback(meshSizeCallback)
    # set off the other options for mesh size determination
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.model.mesh.generate(2)
    rank = MPI.COMM_WORLD.rank
    tmp_folder = pathlib.Path(f"tmp_ellipse_{xrad}_{yrad}_{rank}")
    tmp_folder.mkdir(exist_ok=True)
    gmsh_file = tmp_folder / "ellipse.msh"
    gmsh.write(str(gmsh_file))  # save locally
    gmsh.finalize()

    # return dolfin mesh of max dimension (parent mesh) and marker functions mf1 and mf2
    dmesh, mf1, mf2 = gmsh_to_dolfin(str(gmsh_file), tmp_folder, 2, comm)
    gmsh_file.unlink(missing_ok=False)
    tmp_folder.rmdir()
    return (dmesh, mf1, mf2)


def gmsh_to_dolfin(
    gmsh_file_name: str,
    tmp_folder: pathlib.Path = pathlib.Path("tmp_folder"),
    dimension: int = 3,
    comm: MPI.Comm = d.MPI.comm_world,
) -> Tuple[d.Mesh, d.MeshFunction, d.MeshFunction]:
    """
    Convert .msh file from gmsh to dolfin mesh
    and associated marker files (using meshio).
    Markers are assigned from gmsh mesh, any unassigned
    marker values are given value 0.
    Inputs:
    * gmsh_file_name: .msh file (string)
    * tmp_folder_name: folder name to store temporary mesh files
    * dimension: dimension of parent mesh (int - either 2 or 3)
    Output tuple (dMesh, mf_facet, mf_cell)
    * dMesh: Dolfin-style parent mesh
    * mf_facet: markers for facets
    * mf_cell: markers for cells
    """
    import meshio

    # load, convert to xdmf, and save as temp files
    mesh_in = meshio.read(gmsh_file_name)
    if dimension == 2:
        cell_type = "triangle"
        facet_type = "line"
    elif dimension == 3:
        cell_type = "tetra"
        facet_type = "triangle"
    else:
        ValueError(f"Mesh of dimension {dimension} not implemented")
    # convert cell mesh
    cells = mesh_in.get_cells_type(cell_type)
    cell_data = mesh_in.get_cell_data("gmsh:physical", cell_type)  # extract values of tags
    out_mesh_cell = meshio.Mesh(
        points=mesh_in.points,
        cells={cell_type: cells},
        cell_data={"mf_data": [cell_data]},
    )
    tmp_file_cell = tmp_folder / "tempmesh_cell.xdmf"
    meshio.write(tmp_file_cell, out_mesh_cell)
    # convert facet mesh
    facets = mesh_in.get_cells_type(facet_type)
    facet_data = mesh_in.get_cell_data("gmsh:physical", facet_type)  # extract values of tags
    out_mesh_facet = meshio.Mesh(
        points=mesh_in.points,
        cells={facet_type: facets},
        cell_data={"mf_data": [facet_data]},
    )
    tmp_file_facet = tmp_folder / "tempmesh_facet.xdmf"
    meshio.write(tmp_file_facet, out_mesh_facet)

    # convert xdmf mesh to dolfin-style mesh

    dmesh = d.Mesh(comm)
    mvc_cell = d.MeshValueCollection("size_t", dmesh, dimension)
    with d.XDMFFile(comm, str(tmp_file_cell)) as infile:
        infile.read(dmesh)
        infile.read(mvc_cell, "mf_data")
    mf_cell = d.cpp.mesh.MeshFunctionSizet(dmesh, mvc_cell)
    # set unassigned volumes to tag=0
    mf_cell.array()[np.where(mf_cell.array() > 1e9)[0]] = 0

    mvc_facet = d.MeshValueCollection("size_t", dmesh, dimension - 1)
    with d.XDMFFile(comm, str(tmp_file_facet)) as infile:
        infile.read(mvc_facet, "mf_data")
    mf_facet = d.cpp.mesh.MeshFunctionSizet(dmesh, mvc_facet)
    # set unassigned faces to tag=0
    mf_facet.array()[np.where(mf_facet.array() > 1e9)[0]] = 0

    # remove temp meshes
    tmp_file_cell.unlink(missing_ok=False)
    tmp_file_cell.with_suffix(".h5").unlink(missing_ok=False)
    tmp_file_facet.unlink(missing_ok=False)
    tmp_file_facet.with_suffix(".h5").unlink(missing_ok=False)
    # return dolfin mesh and mfs (marker functions)
    return (dmesh, mf_facet, mf_cell)


def write_mesh(
    mesh: d.Mesh,
    mf2: d.MeshFunction,
    mf3: d.MeshFunction,
    filename: pathlib.Path = pathlib.Path("DemoCuboidMesh.h5"),
):
    # Write mesh and meshfunctions to file
    hdf5 = d.HDF5File(mesh.mpi_comm(), str(filename.with_suffix(".h5")), "w")
    hdf5.write(mesh, "/mesh")
    hdf5.write(mf3, "/mf3")
    hdf5.write(mf2, "/mf2")
    # For visualization of domains
    d.File(str(filename.with_stem(filename.stem + "_mf3").with_suffix(".pvd"))) << mf3
    d.File(str(filename.with_stem(filename.stem + "_mf2").with_suffix(".pvd"))) << mf2
