from typing import Tuple
import os
import pathlib
import numpy as np
import dolfin as d
from mpi4py import MPI

__all__ = [
    "facet_topology",
    "cube_condition",
    "DemoCuboidsMesh",
    "DemoSpheresMesh",
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
    comm: MPI.Comm = d.MPI.comm_world
) -> Tuple[d.Mesh, d.MeshFunction, d.MeshFunction]:
    """
    Creates a mesh for use in examples that contains
    two distinct sphere subvolumes with a shared interface
    surface. If the radius of the inner sphere is 0, mesh a
    single sphere.

    Args:
        outerRad: The radius of the outer sphere
        innerRad: The radius of the inner sphere
        hEdge: maximum mesh size at the outer edge
        hInnerEdge: maximum mesh size at the edge
        of the inner sphere interface_marker: The
        value to mark facets on the interface with
        outer_marker: The value to mark facets on the outer sphere with
        inner_vol_tag: The value to mark the inner spherical volume with
        outer_vol_tag: The value to mark the outer spherical volume with
        comm: MPI communicator to create the mesh with
    Returns:
        A triplet (mesh, facet_marker, cell_marker)
    """
    import gmsh

    assert not np.isclose(outerRad, 0)
    if np.isclose(hEdge, 0):
        hEdge = 0.1 * outerRad
    if np.isclose(hInnerEdge, 0):
        hInnerEdge = 0.2 * outerRad if np.isclose(innerRad, 0) else 0.2 * innerRad
    # Create the two sphere mesh using gmsh
    gmsh.initialize()
    gmsh.model.add("twoSpheres")
    # first add sphere 1 of radius outerRad and center (0,0,0)
    outer_sphere = gmsh.model.occ.addSphere(0, 0, 0, outerRad)
    if np.isclose(innerRad, 0):
        # Use outer_sphere only
        gmsh.model.occ.synchronize()
        gmsh.model.add_physical_group(3, [outer_sphere], tag=outer_vol_tag)
        facets = gmsh.model.getBoundary([(3, outer_sphere)])
        assert len(facets) == 1
        gmsh.model.add_physical_group(2, [facets[0][1]], tag=outer_marker)
    else:
        # Add inner_sphere (radius innerRad, center (0,0,0))
        inner_sphere = gmsh.model.occ.addSphere(0, 0, 0, innerRad)
        # Create interface between spheres
        two_spheres, (outer_sphere_map, inner_sphere_map) = gmsh.model.occ.fragment(
            [(3, outer_sphere)], [(3, inner_sphere)]
        )
        gmsh.model.occ.synchronize()

        # Get the outer boundary
        outer_shell = gmsh.model.getBoundary(two_spheres, oriented=False)
        assert len(outer_shell) == 1
        # Get the inner boundary
        inner_shell = gmsh.model.getBoundary(inner_sphere_map, oriented=False)
        assert len(inner_shell) == 1
        # Add physical markers for facets
        gmsh.model.add_physical_group(outer_shell[0][0], [outer_shell[0][1]], tag=outer_marker)
        gmsh.model.add_physical_group(inner_shell[0][0], [inner_shell[0][1]], tag=interface_marker)

        # Physical markers for
        all_volumes = [tag[1] for tag in outer_sphere_map]
        inner_volume = [tag[1] for tag in inner_sphere_map]
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
        # for one sphere (innerRad = 0), if hEdge > 0.2*outerRad,
        # then lc = 0.2*outerRad in the whole volume
        # for two spheres, if hEdge or hInnerEdge > 0.2*innerRad,
        # they are set to lc = 0.2*innerRad
        R = np.sqrt(x**2 + y**2 + z**2)
        lc1 = hEdge
        lc2 = hInnerEdge
        lc3 = 0.2 * outerRad if np.isclose(innerRad, 0) else 0.2 * innerRad
        if R > innerRad:
            lcTest = lc1 + (lc2 - lc1) * (outerRad - R) / (outerRad - innerRad)
        else:
            lcTest = lc2 + (lc3 - lc2) * (innerRad - R) / innerRad
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
    out_msh_file = pathlib.Path(f"twoSpheres{rank}.msh")

    gmsh.write(str(out_msh_file))  # save locally
    gmsh.finalize()

    import meshio

    # load, convert to xdmf, and save as temp files
    mesh3d_in = meshio.read(out_msh_file)
    out_msh_file.unlink(missing_ok=False)

    def create_mesh(mesh, cell_type):
        cells = mesh.get_cells_type(cell_type)
        cell_data = mesh.get_cell_data("gmsh:physical", cell_type)  # extract values of tags
        out_mesh = meshio.Mesh(
            points=mesh.points,
            cells={cell_type: cells},
            cell_data={"mf_data": [cell_data]},
        )
        return out_mesh

    tet_mesh = create_mesh(mesh3d_in, "tetra")
    tri_mesh = create_mesh(mesh3d_in, "triangle")
    temp_2D = pathlib.Path(f"tempmesh_2dout{rank}.xdmf")
    temp_3D = pathlib.Path(f"tempmesh_3dout{rank}.xdmf")    
    meshio.write(temp_3D, tet_mesh)
    meshio.write(temp_2D, tri_mesh)

    # convert xdmf mesh to dolfin-style mesh
    dmesh = d.Mesh(comm)
    mvc3 = d.MeshValueCollection("size_t", dmesh, 3)
    with d.XDMFFile(comm, str(temp_3D)) as infile:
        infile.read(dmesh)
        infile.read(mvc3, "mf_data")
    mf3 = d.cpp.mesh.MeshFunctionSizet(dmesh, mvc3)
    # set unassigned volumes to tag=0
    mf3.array()[np.where(mf3.array() > 1e9)[0]] = 0
    mvc2 = d.MeshValueCollection("size_t", dmesh, 2)
    with d.XDMFFile(comm, str(temp_2D)) as infile:
        infile.read(mvc2, "mf_data")
    mf2 = d.cpp.mesh.MeshFunctionSizet(dmesh, mvc2)
    # set inner faces to tag=0
    mf2.array()[np.where(mf2.array() > 1e9)[0]] = 0

    # use os to remove temp meshes
    temp_2D.unlink(missing_ok=False)
    temp_3D.unlink(missing_ok=False)
    temp_2D.with_suffix(".h5").unlink(missing_ok=False)
    temp_3D.with_suffix(".h5").unlink(missing_ok=False)

    # return dolfin mesh, mf2 (2d tags) and mf3 (3d tags)
    return (dmesh, mf2, mf3)


def write_mesh(mesh:d.Mesh, mf2:d.MeshFunction, mf3:d.MeshFunction, filename:pathlib.Path=pathlib.Path("DemoCuboidMesh.h5")):
    # Write mesh and meshfunctions to file
    hdf5 = d.HDF5File(mesh.mpi_comm(), str(filename.with_suffix(".h5")), "w")
    hdf5.write(mesh, "/mesh")
    hdf5.write(mf3, "/mf3")
    hdf5.write(mf2, "/mf2")
    # For visualization of domains
    filename.with_stem()
    d.File(str(filename.with_stem(filename.stem+"_mf3").with_suffix(".pvd"))) << mf3
    d.File(str(filename.with_stem(filename.stem+"_mf2").with_suffix(".pvd"))) << mf2
