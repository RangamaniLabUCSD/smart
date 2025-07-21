"""
Functions to create meshes for demos.

:func:`create_cubes` defines a 'cube-in-a-cube' mesh using the
built in :class:`dolfin.UnitCubeMesh()`, with subdomains defined and
marked by :func:`facet_topology` and :func:`cube_condition`
:func:`create_spheres`, :func:`create_ellipsoids`,
:func:`create_cylinders`, and :func:`create_ellipses`
define meshes using gmsh, which are then converted to
dolfin meshes using :func:`gmsh_to_dolfin`
:func:`write_mesh` writes 3d meshes with cell markers ``mf3`` and
facet markers ``mf2`` to hdf5 and pvd files.
"""

from typing import Tuple, NamedTuple, Optional, Union
import pathlib
import numpy as np
import sympy as sym
import dolfin as d
from mpi4py import MPI
from sympy.parsing.sympy_parser import parse_expr
from sympy.solvers.solveset import solveset_real

__all__ = [
    "implicit_curve",
    "facet_topology",
    "cube_condition",
    "create_cubes",
    "create_spheres",
    "create_ellipsoids",
    "create_axisymm",
    "create_cylinders",
    "create_ellipses",
    "compute_curvature",
    "create_2Dcell",
    "gmsh_to_dolfin",
    "write_mesh",
]


def implicit_curve(boundExpr, num_points=51):
    """
    Output (r,z) coordinates given a string expression in the r-z plane
    Currently the function only outputs cell coordinates in the right half
    of the plane (r > 0) for use in axisymmetric formulations.
    If the shape extends to z <= 0, the curve is terminated at precisely z=0,
    which corresponds to a substrate in most cases.
    The (r,z) coordinates start at the maximum z value for r = 0
    and end when either z = 0 or r = 0.
    num_points can be increased for more accuracy in generating the contour,
    but this function will run more slowly for very high num_points.

    Examples:
    * boundExpr = "r**2 + z**2 - 1": defines a quarter circle from (0, 1) to (1, 0)
    * boundExpr = "r**2 + (z-2)**2 - 1": defines a half circle from (0, 3) to (1, 2) to (0, 1)
    """
    outerExprRef = parse_expr(boundExpr)
    r = sym.Symbol("r", real=True)
    z = sym.Symbol("z", real=True)
    outerExpr0 = outerExprRef.subs({"r": 0, "z": z})
    z0 = solveset_real(outerExpr0, z)
    z0Array = np.array([float(z0Val) for z0Val in z0])
    rVals = [0.0]
    zVals = [float(max(z0))]
    rMax = solveset_real(outerExprRef.subs({"r": r, "z": 0.0}), r)
    if len(rMax) > 0:
        sGap = max([float(max(z0)), float(max(rMax))]) / (num_points - 1)
    elif len(z0) == 2 and np.all(z0Array > 0):  # then probably a nuclear contour
        sGap = float(max(z0) - min(z0)) / (num_points - 1)
    else:
        sGap = float(max(z0)) / (num_points - 1)
    curTan = [1, 0]
    while rVals[-1] >= 0 and zVals[-1] >= 0:
        rNext = rVals[-1] + curTan[0] * sGap
        zNext = zVals[-1] + curTan[1] * sGap
        exprCur = parse_expr(boundExpr)
        exprCur = exprCur.subs({"r": rNext, "z": z})
        zNextSol = list(solveset_real(exprCur, z))
        zNextList = [0] * len(zNextSol)
        for i in range(len(zNextSol)):
            zNextList[i] = abs(zNextSol[i] - zNext)
        if zNextSol == [] or min(zNextList) > sGap:
            exprCur = parse_expr(boundExpr)
            exprCur = exprCur.subs({"r": r, "z": zNext})
            rNextSol = list(solveset_real(exprCur, r))
            rNextList = [0] * len(rNextSol)
            for i in range(len(rNextSol)):
                rNextList[i] = abs(rNextSol[i] - rNext)
            idx = rNextList.index(min(rNextList))
            if rNextSol == [] or min(rNextList) > sGap:
                ValueError("Next point could not be found")
            rNext = rNextSol[idx]
        else:
            idx = zNextList.index(min(zNextList))
            zNext = zNextSol[idx]
        rVals.append(float(rNext))
        zVals.append(float(zNext))
        curTan = np.array([rVals[-1] - rVals[-2], zVals[-1] - zVals[-2]])
        curTan = curTan / np.sqrt(float(curTan[0] ** 2 + curTan[1] ** 2))
    if rVals[-1] < 0:
        rVals[-1] = 0
        zVals[-1] = zVals[-2] + (zVals[-1] - zVals[-2]) * (0 - rVals[-2]) / (rVals[-1] - rVals[-2])
    elif zVals[-1] < 0:
        zVals[-1] = 0
        rVals[-1] = rVals[-2] + (rVals[-1] - rVals[-2]) * (0 - zVals[-2]) / (zVals[-1] - zVals[-2])
    rVals = np.array(rVals)
    zVals = np.array(zVals)
    return (rVals, zVals)


def facet_topology(f: d.Facet, mf3: d.MeshFunction):
    """
    Given a facet and cell mesh function,
    return the topology of the face
    as either 'boundary' (outer boundary),
    'internal', or 'interface' (boundary of inner cube)
    """
    # cells adjacent face

    # Initialize facet to cell connectivity
    mf3.mesh().init(2, 3)
    cell_values = mf3.array()
    connected_cells = f.entities(f.mesh().topology().dim())
    if f.exterior():
        return "boundary", cell_values[connected_cells]
    else:
        if len(connected_cells) == 2:
            if cell_values[connected_cells[0]] == cell_values[connected_cells[1]]:
                return "internal", cell_values[connected_cells]
            else:
                return "interface", cell_values[connected_cells]
        else:
            assert len(connected_cells) == 1
            raise Exception(
                "Missing information for determining if facet is internal or interface",
                ", have you turned ghost mode to 'shared_facet'?",
            )


def cube_condition(cell, xmin=0.3, xmax=0.7):
    """
    Returns true when inside an inner cube region defined as:
    xmin <= x <= xmax, xmin <= y <= xmax, xmin <= z <= xmax
    """
    return (
        (xmin - d.DOLFIN_EPS < cell.midpoint().x() < xmax + d.DOLFIN_EPS)
        and (xmin - d.DOLFIN_EPS < cell.midpoint().y() < xmax + d.DOLFIN_EPS)
        and (xmin - d.DOLFIN_EPS < cell.midpoint().z() < xmax + d.DOLFIN_EPS)
    )


def create_cubes(N=16, condition=cube_condition):
    """
    Creates a mesh for use in examples that contains
    two distinct cube subvolumes with a shared interface surface.
    Cell markers:
    1 - Default subvolume (volume outside the inner cube)
    2 - Subvolume specified by condition function

    Facet markers:
    12 - Interface between subvolumes
    10 - Boundary of subvolume 1
    0  - Interior facets
    """
    # Create a mesh (use ghost mode to fix interfaces correctly)
    d.parameters["ghost_mode"] = "shared_facet"
    mesh = d.UnitCubeMesh(N, N, N)
    # Initialize mesh functions
    mf3 = d.MeshFunction("size_t", mesh, 3, 0)
    mf2 = d.MeshFunction("size_t", mesh, 2, 0)

    # Mark all cells that satisfy condition as 3, else 1
    for c in d.cells(mesh, "all"):
        mf3[c] = 2 if condition(c) else 1

    # Mark facets
    for f in d.facets(mesh):
        topology, cellIndices = facet_topology(f, mf3)
        if topology == "interface":
            mf2[f] = 12
        elif topology == "boundary":
            mf2[f] = int(cellIndices[0] * 10)
        else:
            mf2[f] = 0
    return (mesh, mf2, mf3)


def create_spheres(
    outerRad: float = 0.5,
    innerRad: float = 0.25,
    hEdge: float = 0,
    hInnerEdge: float = 0,
    interface_marker: int = 12,
    outer_marker: int = 10,
    inner_vol_tag: int = 2,
    outer_vol_tag: int = 1,
    comm: MPI.Comm = d.MPI.comm_world,
    verbose: bool = False,
) -> Tuple[d.Mesh, d.MeshFunction, d.MeshFunction]:
    """
    Calls create_ellipsoids() to make spherical mesh
    Args:
        outerRad: radius of the outer sphere
        innerRad: radius of the inner sphere
    All other arguments are the same as described for create_ellipsoids()
    """
    dmesh, mf2, mf3 = create_ellipsoids(
        (outerRad, outerRad, outerRad),
        (innerRad, innerRad, innerRad),
        hEdge,
        hInnerEdge,
        interface_marker,
        outer_marker,
        inner_vol_tag,
        outer_vol_tag,
        comm,
        verbose,
    )
    return (dmesh, mf2, mf3)


def create_ellipsoids(
    outerRad: Tuple[float, float, float],
    innerRad: Tuple[float, float, float],
    hEdge: float = 0,
    hInnerEdge: float = 0,
    interface_marker: int = 12,
    outer_marker: int = 10,
    inner_vol_tag: int = 2,
    outer_vol_tag: int = 1,
    comm: MPI.Comm = d.MPI.comm_world,
    verbose: bool = False,
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
            of the inner ellipsoid
        interface_marker: The value to mark facets on the interface with
        outer_marker: The value to mark facets on the outer ellipsoid with
        inner_vol_tag: The value to mark the inner ellipsoidal volume with
        outer_vol_tag: The value to mark the outer ellipsoidal volume with
        comm: MPI communicator to create the mesh with
        verbose: If true print gmsh output, else skip
    Returns:
        Tuple (mesh, facet_marker, cell_marker)
    """

    def meshSizeCallback(dim, tag, x, y, z, lc):
        # mesh length is hEdge at the PM (defaults to 0.1*outerRad,
        # or set when calling function) and hInnerEdge at the ERM
        # (defaults to 0.2*innerRad, or set when calling function)
        # between these, the value is interpolated based on R,
        # and inside the value is interpolated between hInnerEdge and 0.2*innerRad
        # If hInnerEdge > 0.2*innerRad, lc = hInnerRad inside the inner ellipsoid
        # if innerRad=0, then the mesh length is interpolated between
        # hEdge at the PM and 0.2*outerRad in the center
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
            lc3 = max(hInnerEdge, 0.2 * max(innerRad))
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
        return lcTest

    import gmsh

    # Create temporary path for mesh (dependent on number of processes)
    # Only generate mesh on rank 0
    rank = comm.rank
    size = comm.size
    tmp_folder = pathlib.Path(f"tmp_ellipsoid_{outerRad}_{innerRad}_{size}")
    if rank == 0:
        tmp_folder.mkdir(exist_ok=True)
    gmsh_file = tmp_folder / "ellipsoids.msh"

    if rank == 0:
        if np.any(np.isclose(outerRad, 0)):
            ValueError("One of the outer radii is equal to zero")
        if np.isclose(hEdge, 0):
            hEdge = 0.1 * max(outerRad)
        if np.isclose(hInnerEdge, 0):
            hInnerEdge = (
                0.2 * max(outerRad) if np.any(np.isclose(innerRad, 0)) else 0.2 * max(innerRad)
            )
        if innerRad[0] > outerRad[0] or innerRad[1] > outerRad[1] or innerRad[2] > outerRad[2]:
            ValueError("Inner ellipsoid does not fit inside outer ellipsoid")
        # Create the two ellipsoid mesh using gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", int(verbose))

        gmsh.model.add("twoellipsoids")
        # first add ellipsoid 1 of radius outerRad and center (0,0,0)
        outer_ellipsoid = gmsh.model.occ.addSphere(0, 0, 0, 1.0)
        gmsh.model.occ.dilate(
            [(3, outer_ellipsoid)], 0, 0, 0, outerRad[0], outerRad[1], outerRad[2]
        )
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
            gmsh.model.add_physical_group(
                inner_shell[0][0], [inner_shell[0][1]], tag=interface_marker
            )

            # Physical markers for
            all_volumes = [tag[1] for tag in outer_ellipsoid_map]
            inner_volume = [tag[1] for tag in inner_ellipsoid_map]
            outer_volume = []
            for vol in all_volumes:
                if vol not in inner_volume:
                    outer_volume.append(vol)
            gmsh.model.add_physical_group(3, outer_volume, tag=outer_vol_tag)
            gmsh.model.add_physical_group(3, inner_volume, tag=inner_vol_tag)

        gmsh.model.mesh.setSizeCallback(meshSizeCallback)
        # set off the other options for mesh size determination
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        # this changes the algorithm from Frontal-Delaunay to Delaunay,
        # which may provide better results when there are larger gradients in mesh size
        gmsh.option.setNumber("Mesh.Algorithm", 5)

        gmsh.model.mesh.generate(3)
        gmsh.write(str(gmsh_file))
        gmsh.finalize()
    comm.Barrier()
    # return dolfin mesh of max dimension (parent mesh) and marker functions mf2 and mf3
    dmesh, mf2, mf3 = gmsh_to_dolfin(str(gmsh_file), tmp_folder, 3, comm)
    # remove tmp mesh and tmp folder
    if rank == 0:
        gmsh_file.unlink(missing_ok=False)
        tmp_folder.rmdir()

    # return dolfin mesh, mf2 (2d tags) and mf3 (3d tags)
    return (dmesh, mf2, mf3)


def create_axisymm(
    outerExpr: str = "",
    innerExpr: str = "",
    hEdge: float = 0,
    hInnerEdge: float = 0,
    interface_marker: int = 12,
    outer_marker: int = 10,
    inner_vol_tag: int = 2,
    outer_vol_tag: int = 1,
    comm: MPI.Comm = d.MPI.comm_world,
    verbose: bool = False,
) -> Tuple[d.Mesh, d.MeshFunction, d.MeshFunction]:
    """
    Creates an axisymmetric mesh, with the bounding curve defined in
    terms of r and z. (e.g. unit circle defined by "r**2 + (z-1)**2 - 1")
    It is assumed that substrate is present at z = 0, so if the curve extends
    below z = 0 , there is a sharp cutoff.
    Can include one compartment inside another compartment

    Args:
        outerExpr: String implicitly defining an r-z curve for the outer surface
        innerExpr: String implicitly defining an r-z curve for the inner surface
        hEdge: maximum mesh size at the outer edge
        hInnerEdge: maximum mesh size at the edge
            of the inner compartment
        interface_marker: The value to mark facets on the interface with
        outer_marker: The value to mark facets on the outer ellipsoid with
        inner_vol_tag: The value to mark the inner ellipsoidal volume with
        outer_vol_tag: The value to mark the outer ellipsoidal volume with
        comm: MPI communicator to create the mesh with
        verbose: If true print gmsh output, else skip
    Returns:
        Tuple (mesh, facet_marker, cell_marker)
    """
    import gmsh

    if outerExpr == "":
        ValueError("Outer surface is not defined")

    rValsOuter, zValsOuter = implicit_curve(outerExpr)

    if not innerExpr == "":
        rValsInner, zValsInner = implicit_curve(innerExpr)
        zMid = np.mean(zValsInner)
        ROuterVec = np.sqrt(rValsOuter**2 + (zValsOuter - zMid) ** 2)
        RInnerVec = np.sqrt(rValsInner**2 + (zValsInner - zMid) ** 2)
        maxOuterDim = max(ROuterVec)
        maxInnerDim = max(RInnerVec)
    else:
        zMid = np.mean(zValsOuter)
        ROuterVec = np.sqrt(rValsOuter**2 + (zValsOuter - zMid) ** 2)
        maxOuterDim = max(ROuterVec)
    if np.isclose(hEdge, 0):
        hEdge = 0.1 * maxOuterDim
    if np.isclose(hInnerEdge, 0):
        hInnerEdge = 0.2 * maxOuterDim if innerExpr == "" else 0.2 * maxInnerDim
    # Create the two axisymmetric body mesh using gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", int(verbose))
    gmsh.model.add("axisymm")
    # first add outer body and revolve
    outer_tag_list = []
    for i in range(len(rValsOuter)):
        cur_tag = gmsh.model.occ.add_point(rValsOuter[i], 0, zValsOuter[i])
        outer_tag_list.append(cur_tag)
    outer_spline_tag = gmsh.model.occ.add_spline(outer_tag_list)
    if np.isclose(zValsOuter[-1], 0):  # then include substrate at z=0
        origin_tag = gmsh.model.occ.add_point(0, 0, 0)
        symm_axis_tag = gmsh.model.occ.add_line(origin_tag, outer_tag_list[0])
        bottom_tag = gmsh.model.occ.add_line(origin_tag, outer_tag_list[-1])
        outer_loop_tag = gmsh.model.occ.add_curve_loop(
            [outer_spline_tag, symm_axis_tag, bottom_tag]
        )
    else:
        symm_axis_tag = gmsh.model.occ.add_line(outer_tag_list[0], outer_tag_list[-1])
        outer_loop_tag = gmsh.model.occ.add_curve_loop([outer_spline_tag, symm_axis_tag])
    cell_plane_tag = gmsh.model.occ.add_plane_surface([outer_loop_tag])
    outer_shape = gmsh.model.occ.revolve([(2, cell_plane_tag)], 0, 0, 0, 0, 0, 1, 2 * np.pi)
    outer_shape_tags = []
    for i in range(len(outer_shape)):
        if outer_shape[i][0] == 3:  # pull out tags associated with 3d objects
            outer_shape_tags.append(outer_shape[i][1])
    assert len(outer_shape_tags) == 1  # should be just one 3D body from the full revolution

    if innerExpr == "":
        # No inner shape in this case
        gmsh.model.occ.synchronize()
        gmsh.model.add_physical_group(3, outer_shape_tags, tag=outer_vol_tag)
        facets = gmsh.model.getBoundary([(3, outer_shape_tags[0])])
        assert (
            len(facets) == 2
        )  # 2 boundaries because of bottom surface at z = 0, both belong to PM
        gmsh.model.add_physical_group(2, [facets[0][1], facets[1][1]], tag=outer_marker)
    else:
        # Add inner shape
        inner_tag_list = []
        for i in range(len(rValsInner)):
            cur_tag = gmsh.model.occ.add_point(rValsInner[i], 0, zValsInner[i])
            inner_tag_list.append(cur_tag)
        inner_spline_tag = gmsh.model.occ.add_spline(inner_tag_list)
        symm_inner_tag = gmsh.model.occ.add_line(inner_tag_list[0], inner_tag_list[-1])
        inner_loop_tag = gmsh.model.occ.add_curve_loop([inner_spline_tag, symm_inner_tag])
        inner_plane_tag = gmsh.model.occ.add_plane_surface([inner_loop_tag])
        inner_shape = gmsh.model.occ.revolve([(2, inner_plane_tag)], 0, 0, 0, 0, 0, 1, 2 * np.pi)
        inner_shape_tags = []
        for i in range(len(inner_shape)):
            if inner_shape[i][0] == 3:  # pull out tags associated with 3d objects
                inner_shape_tags.append(inner_shape[i][1])
        assert len(inner_shape_tags) == 1  # should be just one 3D body from the full revolution

        # Create interface between 2 objects
        two_shapes, (outer_shape_map, inner_shape_map) = gmsh.model.occ.fragment(
            [(3, outer_shape_tags[0])], [(3, inner_shape_tags[0])]
        )
        gmsh.model.occ.synchronize()

        # Get the outer boundary
        outer_shell = gmsh.model.getBoundary(two_shapes, oriented=False)
        assert (
            len(outer_shell) == 2
        )  # 2 boundaries because of bottom surface at z = 0, both belong to PM
        # Get the inner boundary
        inner_shell = gmsh.model.getBoundary(inner_shape_map, oriented=False)
        assert len(inner_shell) == 1
        # Add physical markers for facets
        gmsh.model.add_physical_group(
            outer_shell[0][0], [outer_shell[0][1], outer_shell[1][1]], tag=outer_marker
        )
        gmsh.model.add_physical_group(inner_shell[0][0], [inner_shell[0][1]], tag=interface_marker)

        # Physical markers for
        all_volumes = [tag[1] for tag in outer_shape_map]
        inner_volume = [tag[1] for tag in inner_shape_map]
        outer_volume = []
        for vol in all_volumes:
            if vol not in inner_volume:
                outer_volume.append(vol)
        gmsh.model.add_physical_group(3, outer_volume, tag=outer_vol_tag)
        gmsh.model.add_physical_group(3, inner_volume, tag=inner_vol_tag)

    def meshSizeCallback(dim, tag, x, y, z, lc):
        # mesh length is hEdge at the PM and hInnerEdge at the inner membrane
        # between these, the value is interpolated based on the relative distance
        # between the two membranes.
        # Inside the inner shape, the value is interpolated between hInnerEdge
        # and lc3, where lc3 = max(hInnerEdge, 0.2*maxInnerDim)
        # if innerRad=0, then the mesh length is interpolated between
        # hEdge at the PM and 0.2*maxOuterDim in the center
        rCur = np.sqrt(x**2 + y**2)
        RCur = np.sqrt(rCur**2 + (z - zMid) ** 2)
        outer_dist = np.sqrt((rCur - rValsOuter) ** 2 + (z - zValsOuter) ** 2)
        np.append(outer_dist, z)  # include the distance from the substrate
        dist_to_outer = min(outer_dist)
        if innerExpr == "":
            lc3 = 0.2 * maxOuterDim
            dist_to_inner = RCur
            in_outer = True
        else:
            inner_dist = np.sqrt((rCur - rValsInner) ** 2 + (z - zValsInner) ** 2)
            dist_to_inner = min(inner_dist)
            inner_idx = np.argmin(inner_dist)
            inner_rad = RInnerVec[inner_idx]
            R_rel_inner = RCur / inner_rad
            lc3 = max(hInnerEdge, 0.2 * maxInnerDim)
            in_outer = R_rel_inner > 1
        lc1 = hEdge
        lc2 = hInnerEdge
        if in_outer:
            lcTest = lc1 + (lc2 - lc1) * (dist_to_outer) / (dist_to_inner + dist_to_outer)
        else:
            lcTest = lc2 + (lc3 - lc2) * (1 - R_rel_inner)
        return lcTest

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
    tmp_folder = pathlib.Path(f"tmp_axisymm_{rank}")
    tmp_folder.mkdir(exist_ok=True)
    gmsh_file = tmp_folder / "axisymm.msh"
    gmsh.write(str(gmsh_file))
    gmsh.finalize()

    # return dolfin mesh of max dimension (parent mesh) and marker functions mf2 and mf3
    dmesh, mf2, mf3 = gmsh_to_dolfin(str(gmsh_file), tmp_folder, 3, comm)
    # remove tmp mesh and tmp folder
    gmsh_file.unlink(missing_ok=False)
    tmp_folder.rmdir()
    # return dolfin mesh, mf2 (2d tags) and mf3 (3d tags)
    return (dmesh, mf2, mf3)


def create_cylinders(
    outerRad: float = 1.0,
    innerRad: float = 0.0,
    outerLength: float = 10.0,
    innerLength: float = 8.0,
    hEdge: float = 0,
    hInnerEdge: float = 0,
    interface_marker: int = 12,
    outer_marker: int = 10,
    inner_vol_tag: int = 2,
    outer_vol_tag: int = 1,
    comm: MPI.Comm = d.MPI.comm_world,
    verbose: bool = False,
) -> Tuple[d.Mesh, d.MeshFunction, d.MeshFunction]:
    """
    Creates a mesh for use in examples that contains
    two distinct ellipsoid subvolumes with a shared interface
    surface. If the radius of the inner ellipsoid is 0, mesh a
    single ellipsoid.

    Args:
        outerRad: The radius of the outer cylinder
        innerRad: The radius of the inner cylinder
        outerLength: length of the outer cylinder
        innerLength: length of the inner cylinder
        hEdge: maximum mesh size at the outer edge
        hInnerEdge: maximum mesh size at the edge
            of the inner cylinder
        interface_marker: The value to mark facets on the interface with
        outer_marker: The value to mark facets on the outer ellipsoid with
        inner_vol_tag: The value to mark the inner spherical volume with
        outer_vol_tag: The value to mark the outer spherical volume with
        comm: MPI communicator to create the mesh with
        verbose: If true print gmsh output, else skip
    Returns:
        A triplet (mesh, facet_marker, cell_marker)
    """
    import gmsh

    if np.isclose(outerRad, 0):
        ValueError("Outer radius is equal to zero")
    if np.isclose(hEdge, 0):
        hEdge = 0.1 * max(outerRad)
    if np.isclose(hInnerEdge, 0):
        hInnerEdge = 0.2 * outerRad if np.isclose(innerRad, 0) else 0.2 * innerRad
    if innerRad > outerRad or innerLength >= outerLength:
        ValueError("Inner cylinder does not fit inside outer cylinder")
    # Create the two cylinder mesh using gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", int(verbose))

    gmsh.model.add("twocylinders")
    # first add cylinder 1 of radius outerRad and center (0,0,0)
    outer_cylinder = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, outerLength, outerRad)
    if np.isclose(innerRad, 0):
        # Use outer_cylinder only
        gmsh.model.occ.synchronize()
        gmsh.model.add_physical_group(3, [outer_cylinder], tag=outer_vol_tag)
        facets = gmsh.model.getBoundary([(3, outer_cylinder)])
        gmsh.model.add_physical_group(2, [facets[0][1]], tag=outer_marker)
    else:
        # Add inner_cylinder (radius innerRad,
        # center (0,0,(outerLength-innerLength)/2))
        inner_cylinder = gmsh.model.occ.addCylinder(
            0, 0, (outerLength - innerLength) / 2, 0, 0, innerLength, innerRad
        )
        # Create interface between cylinders
        two_cylinders, (outer_cylinder_map, inner_cylinder_map) = gmsh.model.occ.fragment(
            [(3, outer_cylinder)], [(3, inner_cylinder)]
        )
        gmsh.model.occ.synchronize()

        # Get the outer boundary
        outer_shell = gmsh.model.getBoundary(two_cylinders, oriented=False)
        # Get the inner boundary
        inner_shell = gmsh.model.getBoundary(inner_cylinder_map, oriented=False)
        # Add physical markers for facets
        gmsh.model.add_physical_group(outer_shell[0][0], [outer_shell[0][1]], tag=outer_marker)
        gmsh.model.add_physical_group(inner_shell[0][0], [inner_shell[0][1]], tag=interface_marker)

        # Physical markers for
        all_volumes = [tag[1] for tag in outer_cylinder_map]
        inner_volume = [tag[1] for tag in inner_cylinder_map]
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
        # between these, the value is interpolated based on r (polar coord),
        # and inside the value is interpolated between hInnerEdge and 0.2*innerRad
        # if innerRad=0, then the mesh length is interpolated between
        # hEdge at the PM and 0.2*outerRad in the center
        r_cur = np.sqrt(x**2 + y**2)
        if np.isclose(innerRad, 0):
            lc3 = 0.2 * outerRad
        else:
            lc3 = max(hInnerEdge, 0.2 * innerRad)
        lc1 = hEdge
        lc2 = hInnerEdge
        if r_cur > innerRad:
            lcTest = lc1 + (lc2 - lc1) * (outerRad - r_cur) / (outerRad - innerRad)
        else:
            lcTest = lc2 + (lc3 - lc2) * (innerRad - r_cur) / innerRad
        return lcTest

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
    tmp_folder = pathlib.Path(f"tmp_cylinder_{outerRad}_{innerRad}_{rank}")
    tmp_folder.mkdir(exist_ok=True)
    gmsh_file = tmp_folder / "cylinders.msh"
    gmsh.write(str(gmsh_file))
    gmsh.finalize()

    # return dolfin mesh of max dimension (parent mesh) and marker functions mf2 and mf3
    dmesh, mf2, mf3 = gmsh_to_dolfin(str(gmsh_file), tmp_folder, 3, comm)
    # remove tmp mesh and tmp folder
    gmsh_file.unlink(missing_ok=False)
    tmp_folder.rmdir()
    # return dolfin mesh, mf2 (2d tags) and mf3 (3d tags)
    return (dmesh, mf2, mf3)


def create_multicell(
    cubeSize: float = 100.0,
    locVec1: list = [[0, 0, 0]],
    locVec2: list = [],
    cellRad1: float = 10.0,
    cellRad2: float = 10.0,
    hCube: float = 0,
    hCell: float = 0,
    interface_marker1: int = 11,
    interface_marker2: int = 12,
    outer_marker: int = 10,
    extracell_tag: int = 1,
    cell_vol_tag1: int = 2,
    cell_vol_tag2: int = 3,
    comm: MPI.Comm = d.MPI.comm_world,
    verbose: bool = False,
) -> Tuple[d.Mesh, d.MeshFunction, d.MeshFunction]:
    """
    Creates a mesh with an outer cube containing embedded cells at specified locations.
    Args:
        cubeSize: Length of cube sides
        locVec: vector of cell locations
        hCube: maximum mesh size for cube
        hCell: maximum mesh size for cell surfaces
        interface_marker: The value to mark facets on the interface with
        outer_marker: The value to mark facets on the outer ellipsoid with
        inner_vol_tag: The value to mark the inner spherical volume with
        outer_vol_tag: The value to mark the outer spherical volume with
        comm: MPI communicator to create the mesh with
        verbose: If true print gmsh output, else skip
    Returns:
        A triplet (mesh, facet_marker, cell_marker)
    """
    import gmsh

    if np.isclose(cubeSize, 0):
        ValueError("Outer cube size is equal to zero")
    if np.isclose(hCube, 0):
        hCube = 0.1 * max(cubeSize)
    if np.isclose(hCell, 0):
        hCell = 0.2 * cubeSize if np.isclose(cellRad1, 0) else 0.2 * cellRad1
    # if innerRad > outerRad or innerLength >= outerLength:
    #     ValueError("Inner cylinder does not fit inside outer cylinder")
    # Create the two cylinder mesh using gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", int(verbose))

    gmsh.model.add("multicell")
    # first add outer cube
    cube = gmsh.model.occ.addBox(
        -cubeSize / 2, -cubeSize / 2, -cubeSize / 2, cubeSize, cubeSize, cubeSize
    )
    if (np.isclose(cellRad1, 0) or len(locVec1) == 0) and (
        np.isclose(cellRad2, 0) or len(locVec2) == 0
    ):
        # Just a cube!
        gmsh.model.occ.synchronize()
        gmsh.model.add_physical_group(3, [cube], tag=extracell_tag)
        facets = gmsh.model.getBoundary([(3, cube)])
        gmsh.model.add_physical_group(2, [facets[0][1]], tag=outer_marker)
    else:
        # Add cells
        cell_list = []
        # first add source cell(s)
        for i in range(len(locVec1)):
            cur_tag = gmsh.model.occ.addSphere(
                locVec1[i][0], locVec1[i][1], locVec1[i][2], cellRad1
            )
            cell_list.append((3, cur_tag))
        # now add additional cells
        for i in range(len(locVec2)):
            cur_tag = gmsh.model.occ.addSphere(
                locVec2[i][0], locVec2[i][1], locVec2[i][2], cellRad2
            )
            cell_list.append((3, cur_tag))
        # Create interface between cells and extracell
        full_geo, maps = gmsh.model.occ.fragment([(3, cube)], cell_list)
        cube_map = maps[0]
        cell_maps = maps[1:]
        gmsh.model.occ.synchronize()

        # Get the outer boundary
        outer_shells = gmsh.model.getBoundary(full_geo, oriented=False)
        # Get the inner boundary
        inner_shells1 = []
        inner_shells2 = []
        for i in range(0, len(locVec1)):
            inner_shells1.append(gmsh.model.getBoundary(cell_maps[i], oriented=False))
        for i in range(len(locVec1), len(cell_maps)):
            inner_shells2.append(gmsh.model.getBoundary(cell_maps[i], oriented=False))
        # Add physical markers for facets
        gmsh.model.add_physical_group(2, [faces[1] for faces in outer_shells], tag=outer_marker)
        gmsh.model.add_physical_group(
            2, [faces[0][1] for faces in inner_shells1], tag=interface_marker1
        )
        gmsh.model.add_physical_group(
            2, [faces[0][1] for faces in inner_shells2], tag=interface_marker2
        )

        # Physical markers for
        all_volumes = [tag[1] for tag in cube_map]
        inner_volumes1 = [tag[0][1] for tag in cell_maps[0 : len(locVec1)]]
        inner_volumes2 = [tag[0][1] for tag in cell_maps[len(locVec1) :]]
        outer_volume = []
        for vol in all_volumes:
            if (vol not in inner_volumes1) and (vol not in inner_volumes2):
                outer_volume.append(vol)
        gmsh.model.add_physical_group(3, outer_volume, tag=extracell_tag)
        gmsh.model.add_physical_group(3, inner_volumes1, tag=cell_vol_tag1)
        gmsh.model.add_physical_group(3, inner_volumes2, tag=cell_vol_tag2)

    def meshSizeCallback(dim, tag, x, y, z, lc):
        # mesh length is hEdge at the PM (defaults to 0.1*outerRad,
        # or set when calling function) and hInnerEdge at the ERM
        # (defaults to 0.2*innerRad, or set when calling function)
        # between these, the value is interpolated based on r (polar coord),
        # and inside the value is interpolated between hInnerEdge and 0.2*innerRad
        # if innerRad=0, then the mesh length is interpolated between
        # hEdge at the PM and 0.2*outerRad in the center

        if (np.isclose(cellRad1, 0) or len(locVec1) == 0) and (
            np.isclose(cellRad2, 0) or len(locVec2) == 0
        ):
            return hCube
        elif np.isclose(cellRad1, 0) or len(locVec1) == 0:
            cell_locs1 = [np.inf]
            cell_locs2 = np.sqrt(
                (x - np.array(locVec2)[:, 0]) ** 2
                + (y - np.array(locVec2)[:, 1]) ** 2
                + (z - np.array(locVec2)[:, 2]) ** 2
            )
        elif np.isclose(cellRad2, 0) or len(locVec2) == 0:
            cell_locs1 = np.sqrt(
                (x - np.array(locVec1)[:, 0]) ** 2
                + (y - np.array(locVec1)[:, 1]) ** 2
                + (z - np.array(locVec1)[:, 2]) ** 2
            )
            cell_locs2 = [np.inf]
        else:
            cell_locs1 = np.sqrt(
                (x - np.array(locVec1)[:, 0]) ** 2
                + (y - np.array(locVec1)[:, 1]) ** 2
                + (z - np.array(locVec1)[:, 2]) ** 2
            )
            cell_locs2 = np.sqrt(
                (x - np.array(locVec2)[:, 0]) ** 2
                + (y - np.array(locVec2)[:, 1]) ** 2
                + (z - np.array(locVec2)[:, 2]) ** 2
            )
        closest_cell1 = min(cell_locs1)
        closest_cell2 = min(cell_locs2)
        if (closest_cell1 < cellRad1) or (closest_cell2 < cellRad2):
            return hCell
        else:
            if closest_cell1 < closest_cell2:
                cellWeight = np.exp(-(closest_cell1 - cellRad1) / (0.2 * cellRad1))
            else:
                cellWeight = np.exp(-(closest_cell2 - cellRad2) / (0.2 * cellRad2))
            return hCell * cellWeight + hCube * (1 - cellWeight)

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
    tmp_folder = pathlib.Path(f"tmp_extracell_{cubeSize}_{cellRad1}_{cellRad2}_{rank}")
    tmp_folder.mkdir(exist_ok=True)
    gmsh_file = tmp_folder / "extracell.msh"
    gmsh.write(str(gmsh_file))
    gmsh.finalize()

    # return dolfin mesh of max dimension (parent mesh) and marker functions mf2 and mf3
    dmesh, mf2, mf3 = gmsh_to_dolfin(str(gmsh_file), tmp_folder, 3, comm)
    # remove tmp mesh and tmp folder
    gmsh_file.unlink(missing_ok=False)
    tmp_folder.rmdir()
    # return dolfin mesh, mf2 (2d tags) and mf3 (3d tags)
    return (dmesh, mf2, mf3)


def create_ellipses(
    xrad_outer: float = 3.0,
    yrad_outer: float = 1.0,
    xrad_inner: float = 0.0,
    yrad_inner: float = 0.0,
    hEdge: float = 0.1,
    hInnerEdge: float = 0.1,
    interface_marker: int = 12,
    outer_marker: int = 10,
    inner_tag: int = 2,
    outer_tag: int = 1,
    comm: MPI.Comm = d.MPI.comm_world,
    verbose: bool = True,
) -> Tuple[d.Mesh, d.MeshFunction, d.MeshFunction]:
    """
    Creates a mesh for an ellipse surface,
    optionally includes an ellipse within the main ellipse
    Args:
        xrad_outer: outer radius assoc with x axis
        yrad_outer: outer radius assoc with y axis
        xrad_inner: x radius of inner ellipse (optional)
        yrad_inner: y radius of inner ellipse (optional)
        hEdge: mesh resolution at outer edge
        hInnerEdge: mesh resolution at edge of inner ellipse
        interface_marker: The value to mark facets on the interface with
        outer_marker: The value to mark facets on edge of the outer ellipse with
        inner_tag: The value to mark the inner ellipse surface with
        outer_tag: The value to mark the outer ellipse surface with
        comm: MPI communicator to create the mesh with
        verbose: If true print gmsh output, else skip
    Returns:
        Tuple (mesh, facet_marker (mf1), cell_marker(mf2))
    """
    import gmsh

    outerRad = [xrad_outer, yrad_outer]
    innerRad = [xrad_inner, yrad_inner]
    if np.any(np.isclose(outerRad, 0)):
        ValueError("One of the outer radii is equal to zero")
    if np.isclose(hEdge, 0):
        hEdge = 0.1 * max(outerRad)
    if np.isclose(hInnerEdge, 0):
        hInnerEdge = 0.2 * max(outerRad) if np.any(np.isclose(innerRad, 0)) else 0.2 * max(innerRad)
    if innerRad[0] > outerRad[0] or innerRad[1] > outerRad[1]:
        ValueError("Inner ellipse does not fit inside outer ellipse")
    # Create the two ellipse mesh using gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", int(verbose))

    gmsh.model.add("ellipses")
    # first add ellipse 1 of radius outerRad and center (0,0,0)
    outer_ellipse = gmsh.model.occ.addDisk(0, 0, 0, xrad_outer, yrad_outer)
    if np.any(np.isclose(innerRad, 0)):
        # Use outer_ellipse only
        gmsh.model.occ.synchronize()
        gmsh.model.add_physical_group(2, [outer_ellipse], tag=outer_tag)
        facets = gmsh.model.getBoundary([(2, outer_ellipse)])
        assert len(facets) == 1
        gmsh.model.add_physical_group(1, [facets[0][1]], tag=outer_marker)
    else:
        # Add inner_ellipse (radius innerRad, center (0,0,0))
        inner_ellipse = gmsh.model.occ.addDisk(0, 0, 0, xrad_inner, yrad_inner)
        # Create interface between ellipses
        two_ellipses, (outer_ellipse_map, inner_ellipse_map) = gmsh.model.occ.fragment(
            [(2, outer_ellipse)], [(2, inner_ellipse)]
        )
        gmsh.model.occ.synchronize()

        # Get the outer boundary
        outer_shell = gmsh.model.getBoundary(two_ellipses, oriented=False)
        assert len(outer_shell) == 1
        # Get the inner boundary
        inner_shell = gmsh.model.getBoundary(inner_ellipse_map, oriented=False)
        assert len(inner_shell) == 1
        # Add physical markers for facets
        gmsh.model.add_physical_group(outer_shell[0][0], [outer_shell[0][1]], tag=outer_marker)
        gmsh.model.add_physical_group(inner_shell[0][0], [inner_shell[0][1]], tag=interface_marker)

        # Physical markers for
        all_surfs = [tag[1] for tag in outer_ellipse_map]
        inner_surf = [tag[1] for tag in inner_ellipse_map]
        outer_surf = []
        for surf in all_surfs:
            if surf not in inner_surf:
                outer_surf.append(surf)
        gmsh.model.add_physical_group(2, outer_surf, tag=outer_tag)
        gmsh.model.add_physical_group(2, inner_surf, tag=inner_tag)

    def meshSizeCallback(dim, tag, x, y, z, lc):
        # mesh length is hEdge at the PM (defaults to 0.1*outerRad,
        # or set when calling function) and hInnerEdge at the ERM
        # (defaults to 0.2*innerRad, or set when calling function)
        # between these, the value is interpolated based on R,
        # and inside the value is interpolated between hInnerEdge and 0.2*innerRad
        # If hInnerEdge > 0.2*innerRad, lc = hInnerEdge inside the inner ellipse
        # if innerRad=0, then the mesh length is interpolated between
        # hEdge at the PM and 0.2*outerRad in the center
        R_rel_outer = np.sqrt((x / outerRad[0]) ** 2 + (y / outerRad[1]) ** 2)
        if np.any(np.isclose(innerRad, 0)):
            lc3 = 0.2 * max(outerRad)
            innerRad_scale = 0
            in_outer = True
        else:
            R_rel_inner = np.sqrt((x / innerRad[0]) ** 2 + (y / innerRad[1]) ** 2)
            lc3 = max(hInnerEdge, 0.2 * max(innerRad))
            innerRad_scale = np.mean([innerRad[0] / outerRad[0], innerRad[1] / outerRad[1]])
            in_outer = R_rel_inner > 1
        lc1 = hEdge
        lc2 = hInnerEdge
        if in_outer:
            lcTest = lc1 + (lc2 - lc1) * (1 - R_rel_outer) / (1 - innerRad_scale)
        else:
            lcTest = lc2 + (lc3 - lc2) * (1 - R_rel_inner)
        return lcTest

    gmsh.model.mesh.setSizeCallback(meshSizeCallback)
    # set off the other options for mesh size determination
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    # this changes the algorithm from Frontal-Delaunay to Delaunay,
    # which may provide better results when there are larger gradients in mesh size
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    gmsh.model.mesh.generate(2)
    rank = MPI.COMM_WORLD.rank
    tmp_folder = pathlib.Path(f"tmp_ellipse_{outerRad}_{innerRad}_{rank}")
    tmp_folder.mkdir(exist_ok=True)
    gmsh_file = tmp_folder / "ellipses.msh"
    gmsh.write(str(gmsh_file))
    gmsh.finalize()

    # return dolfin mesh of max dimension (parent mesh) and marker functions mf2 and mf3
    dmesh, mf1, mf2 = gmsh_to_dolfin(str(gmsh_file), tmp_folder, 2, comm)
    # remove tmp mesh and tmp folder
    gmsh_file.unlink(missing_ok=False)
    tmp_folder.rmdir()
    # return dolfin mesh, mf1 (1d tags) and mf2 (2d tags)
    return (dmesh, mf1, mf2)


def compute_curvature(
    ref_mesh: d.mesh,
    mf_facet: d.MeshFunction,
    mf_cell: d.MeshFunction,
    facet_marker_vec: list,
    cell_marker_vec: list,
    half_mesh_data: Tuple[d.Mesh, d.MeshFunction] = "",
    axisymm: bool = False,
):
    """
    Use dolfin functions to estimate curvature on boundary mesh.
    Boundary meshes are created by extracting the Meshview associated
    with each facet marker value in facet_marker_vec.
    The length of cell_marker_vec must be the same length as facet_marker_vec,
    with each value in the list identifying the marker value for a domain that
    contains the boundary identified by the associated facet_marker.
    For instance, if facet_marker_vec=[10,12] and cell_marker_vec=[1,2], then
    the domain over which mf_cell=1 must contain the domain mf_facet=10 on its boundary
    and the domain mf_cell=2 must contain domain mf_facet=12 on its boundary.
    Mean curvature is approximated by first projecting facet normals (n) onto a boundary
    finite element space and then solving a variational form of kappa = -div(n).

    Args:
        ref_mesh: dolfin mesh describing the entire geometry (parent mesh)
        mf_facet: facet mesh function with boundary domain markers
        mf_cell: cell mesh function with cell domain markers
        facet_marker_vec: list with values of facet markers to iterate over
        cell_marker_vec: list with values of cell markers (see above)
        half_mesh_data: tuple with dolfin mesh for half domain and
                        facet mesh function over half domain. If not specified,
                        this is empty and the curvature values are not mapped
                        onto the half domain
    Returns:
        kappa_mf: Vertex mesh function containing mean curvature values
    """

    if len(facet_marker_vec) != len(cell_marker_vec):
        ValueError("Each listed facet marker value must have an associated cell marker value")

    if half_mesh_data == "":
        kappa_mf = d.MeshFunction("double", ref_mesh, 0)
    else:
        (dmesh_half, mf2_half) = half_mesh_data
        kappa_mf = d.MeshFunction("double", dmesh_half, 0)

    for i in range(len(cell_marker_vec)):
        mesh = d.MeshView.create(mf_cell, cell_marker_vec[i])
        bmesh = d.MeshView.create(mf_facet, facet_marker_vec[i])

        n = d.FacetNormal(mesh)
        # estimate facet normals in CG2 for better smoothness and accuracy
        V = d.VectorFunctionSpace(mesh, "CG", 2)
        u = d.TrialFunction(V)
        v = d.TestFunction(V)
        ds = d.Measure("ds", mesh)
        a = d.inner(u, v) * ds
        lform = d.inner(n, v) * ds
        A = d.assemble(a, keep_diagonal=True)
        L = d.assemble(lform)

        A.ident_zeros()
        nh = d.Function(V)
        d.solve(A, nh.vector(), L)  # project facet normals onto CG1

        Vb = d.FunctionSpace(bmesh, "CG", 1)
        Vb_vec = d.VectorFunctionSpace(bmesh, "CG", 1)
        nb = d.interpolate(nh, Vb_vec)
        p, q = d.TrialFunction(Vb), d.TestFunction(Vb)
        dx = d.Measure("dx", bmesh)
        a = d.inner(p, q) * dx
        lform = d.inner(d.div(nb), q) * dx
        A = d.assemble(a, keep_diagonal=True)
        L = d.assemble(lform)
        A.ident_zeros()
        kappab = d.Function(Vb)
        d.solve(A, kappab.vector(), L)
        if axisymm:  # then include out of plane (parallel) curvature as well
            kappab_vec = kappab.vector().get_local()
            nb_vec = nb.vector().get_local()
            nb_vec = nb_vec.reshape((int(len(nb_vec) / 3), 3))
            normal_angle = np.arctan2(nb_vec[:, 0], nb_vec[:, 2])
            x = Vb.tabulate_dof_coordinates()
            parallel_curv = np.sin(normal_angle) / x[:, 0]
            inf_logic = np.isinf(parallel_curv)
            parallel_curv[inf_logic] = kappab_vec[inf_logic]
            kappab.vector().set_local((kappab_vec + parallel_curv) / 2.0)
            kappab.vector().apply("insert")
        elif ref_mesh.topology().dim() == 3:
            kappab_vec = kappab.vector().get_local()
            kappab.vector().set_local(kappab_vec / 2)
            kappab.vector().apply("insert")

        # map kappab to mesh function
        if half_mesh_data == "":
            store_map_b = bmesh.topology().mapping()[ref_mesh.id()].vertex_map()
            for j in range(len(store_map_b)):
                cur_sub_idx = d.vertex_to_dof_map(Vb)[j]
                kappa_mf.set_value(store_map_b[j], kappab.vector().get_local()[cur_sub_idx])
        else:
            # extract coordinates and kappa values cell-wise from kappab and bmesh
            bmesh_half = d.MeshView.create(mf2_half, facet_marker_vec[i])
            Vb_half = d.FunctionSpace(bmesh_half, "CG", 1)
            kappab_vals = kappab.vector().get_local()
            kappab_cell_vals = []
            kappab_cell_coords = []
            cnum = 0
            for c in d.cells(bmesh):
                cur_idx = Vb.dofmap().cell_dofs(cnum)
                kappab_cell_vals.append(kappab_vals[cur_idx])
                kappab_cell_coords.append(c.get_coordinate_dofs())
                cnum = cnum + 1
            # convert to numpy arrays for interpolation
            kappab_cell_coords = np.array(kappab_cell_coords)
            kappab_cell_vals = np.array(kappab_cell_vals)
            # initialize kappab_half data
            kappab_half_vals = []
            kappab_half_coords = np.array(Vb_half.tabulate_dof_coordinates())
            dthresh1 = 1.5 * ref_mesh.hmax()
            dthresh2 = ref_mesh.hmin() / 10

            # linear interpolation onto half mesh
            for j in range(len(kappab_half_coords)):
                assigned_val = False
                cur_coord = kappab_half_coords[j]
                dxi = cur_coord[0] - kappab_cell_coords[:, 0]
                dzi = cur_coord[2] - kappab_cell_coords[:, 2]
                for k in range(len(dxi)):
                    if np.sqrt(dxi[k] ** 2 + dzi[k] ** 2) > dthresh1:
                        continue
                    else:
                        dx = kappab_cell_coords[k, 3] - kappab_cell_coords[k, 0]
                        dz = kappab_cell_coords[k, 5] - kappab_cell_coords[k, 2]
                        ds = np.sqrt(dx**2 + dz**2)
                        alpha_cur = (dzi[k] * dz + dxi[k] * dx) / ds**2
                        d_cur = np.sqrt(
                            ((dzi[k] * dx * dz - dxi[k] * dz**2) / ds**2) ** 2
                            + ((dxi[k] * dx * dz - dzi[k] * dx**2) / ds**2) ** 2
                        )
                        cur_vals = kappab_cell_vals[k]
                        if alpha_cur >= 0 and alpha_cur <= 1 and d_cur < dthresh2:
                            kappab_half_vals.append(
                                cur_vals[0] + alpha_cur * (cur_vals[1] - cur_vals[0])
                            )
                            assigned_val = True
                            break
                    if not assigned_val:
                        ValueError("Mapping curvatures onto half domain failed")
            # store values in mesh function
            store_map_pm = bmesh_half.topology().mapping()[dmesh_half.id()].vertex_map()
            for j in range(len(store_map_pm)):
                cur_sub_idx = d.vertex_to_dof_map(Vb_half)[j]
                kappa_mf.set_value(store_map_pm[j], kappab_half_vals[cur_sub_idx])
    return kappa_mf


def create_2Dcell(
    outerExpr: str = "",
    innerExpr: str = "",
    hEdge: float = 0,
    hInnerEdge: float = 0,
    interface_marker: int = 12,
    outer_marker: int = 10,
    inner_tag: int = 2,
    outer_tag: int = 1,
    comm: MPI.Comm = d.MPI.comm_world,
    verbose: bool = False,
    half_cell: bool = True,
    return_curvature: bool = False,
) -> Tuple[d.Mesh, d.MeshFunction, d.MeshFunction]:
    """
    Creates a 2D mesh of a cell profile, with the bounding curve defined in
    terms of r and z (e.g. unit circle would be "r**2 + (z-1)**2 - 1)
    It is assumed that substrate is present at z = 0, so if the curve extends
    below z = 0 , there is a sharp cutoff.
    If half_cell = True, only have of the contour is constructed, with a
    left zero-flux boundary at r = 0.
    Can include one compartment inside another compartment.
    Recommended for use with the axisymmetric feature of SMART.

    Args:
        outerExpr: String implicitly defining an r-z curve for the outer surface
        innerExpr: String implicitly defining an r-z curve for the inner surface
        hEdge: maximum mesh size at the outer edge
        hInnerEdge: maximum mesh size at the edge
            of the inner compartment
        interface_marker: The value to mark facets on the interface with
        outer_marker: The value to mark facets on edge of the outer ellipse with
        inner_tag: The value to mark the inner ellipse surface with
        outer_tag: The value to mark the outer ellipse surface with
        comm: MPI communicator to create the mesh with
        verbose: If true print gmsh output, else skip
        half_cell: If true, consider r=0 the symmetry axis for an axisymm shape
    Returns:
        Tuple (mesh, facet_marker, cell_marker)
    """
    import gmsh

    if outerExpr == "":
        ValueError("Outer surface is not defined")

    if return_curvature:
        # create full mesh for curvature analysis and then map onto half mesh
        # if half_cell_with_curvature is True
        half_cell_with_curvature = half_cell
        half_cell = False

    rValsOuter, zValsOuter = implicit_curve(outerExpr)

    if not innerExpr == "":
        rValsInner, zValsInner = implicit_curve(innerExpr)
        zMid = np.mean(zValsInner)
        ROuterVec = np.sqrt(rValsOuter**2 + (zValsOuter - zMid) ** 2)
        RInnerVec = np.sqrt(rValsInner**2 + (zValsInner - zMid) ** 2)
        maxOuterDim = max(ROuterVec)
        maxInnerDim = max(RInnerVec)
    else:
        zMid = np.mean(zValsOuter)
        ROuterVec = np.sqrt(rValsOuter**2 + (zValsOuter - zMid) ** 2)
        maxOuterDim = max(ROuterVec)
    if np.isclose(hEdge, 0):
        hEdge = 0.1 * maxOuterDim
    if np.isclose(hInnerEdge, 0):
        hInnerEdge = 0.2 * maxOuterDim if innerExpr == "" else 0.2 * maxInnerDim
    # Create the 2D mesh using gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", int(verbose))
    gmsh.model.add("2DCell")
    # first add outer body and revolve
    outer_tag_list = []
    for i in range(len(rValsOuter)):
        cur_tag = gmsh.model.occ.add_point(rValsOuter[i], 0, zValsOuter[i])
        outer_tag_list.append(cur_tag)
    outer_spline_tag = gmsh.model.occ.add_spline(outer_tag_list)
    if not half_cell:
        outer_tag_list2 = []
        for i in range(len(rValsOuter)):
            cur_tag = gmsh.model.occ.add_point(-rValsOuter[i], 0, zValsOuter[i])
            outer_tag_list2.append(cur_tag)
        outer_spline_tag2 = gmsh.model.occ.add_spline(outer_tag_list2)
    if np.isclose(zValsOuter[-1], 0):  # then include substrate at z=0
        if half_cell:
            origin_tag = gmsh.model.occ.add_point(0, 0, 0)
            symm_axis_tag = gmsh.model.occ.add_line(origin_tag, outer_tag_list[0])
            bottom_tag = gmsh.model.occ.add_line(origin_tag, outer_tag_list[-1])
            outer_loop_tag = gmsh.model.occ.add_curve_loop(
                [outer_spline_tag, bottom_tag, symm_axis_tag]
            )
        else:
            bottom_tag = gmsh.model.occ.add_line(outer_tag_list[-1], outer_tag_list2[-1])
            outer_loop_tag = gmsh.model.occ.add_curve_loop(
                [outer_spline_tag, outer_spline_tag2, bottom_tag]
            )
    else:
        if half_cell:
            symm_axis_tag = gmsh.model.occ.add_line(outer_tag_list[0], outer_tag_list[-1])
            outer_loop_tag = gmsh.model.occ.add_curve_loop([outer_spline_tag, symm_axis_tag])
        else:
            outer_loop_tag = gmsh.model.occ.add_curve_loop([outer_spline_tag, outer_spline_tag2])
    cell_plane_tag = gmsh.model.occ.add_plane_surface([outer_loop_tag])

    if innerExpr == "":
        # No inner shape in this case
        gmsh.model.occ.synchronize()
        gmsh.model.add_physical_group(2, [cell_plane_tag], tag=outer_tag)
        facets = gmsh.model.getBoundary([(2, cell_plane_tag)])
        facet_tag_list = []
        for i in range(len(facets)):
            facet_tag_list.append(facets[i][1])
        if half_cell:  # if half, set symmetry axis to 0 (no flux)
            xmin, ymin, zmin = (-hInnerEdge / 10, -hInnerEdge / 10, -1)
            xmax, ymax, zmax = (hInnerEdge / 10, hInnerEdge / 10, max(zValsOuter) + 1)
            all_symm_bound = gmsh.model.occ.get_entities_in_bounding_box(
                xmin, ymin, zmin, xmax, ymax, zmax, dim=1
            )
            symm_bound_markers = []
            for i in range(len(all_symm_bound)):
                symm_bound_markers.append(all_symm_bound[i][1])
            gmsh.model.add_physical_group(1, symm_bound_markers, tag=0)
        gmsh.model.add_physical_group(1, facet_tag_list, tag=outer_marker)
    else:
        # Add inner shape
        inner_tag_list = []
        for i in range(len(rValsInner)):
            cur_tag = gmsh.model.occ.add_point(rValsInner[i], 0, zValsInner[i])
            inner_tag_list.append(cur_tag)
        inner_spline_tag = gmsh.model.occ.add_spline(inner_tag_list)
        if half_cell:
            symm_inner_tag = gmsh.model.occ.add_line(inner_tag_list[0], inner_tag_list[-1])
            inner_loop_tag = gmsh.model.occ.add_curve_loop([inner_spline_tag, symm_inner_tag])
        else:
            inner_tag_list2 = []
            for i in range(len(rValsInner)):
                cur_tag = gmsh.model.occ.add_point(-rValsInner[i], 0, zValsInner[i])
                inner_tag_list2.append(cur_tag)
            inner_spline_tag2 = gmsh.model.occ.add_spline(inner_tag_list2)
            inner_loop_tag = gmsh.model.occ.add_curve_loop([inner_spline_tag, inner_spline_tag2])
        inner_plane_tag = gmsh.model.occ.add_plane_surface([inner_loop_tag])
        cell_plane_list = [cell_plane_tag]
        inner_plane_list = [inner_plane_tag]

        outer_volume = []
        inner_volume = []
        all_volumes = []
        inner_marker_list = []
        outer_marker_list = []
        for i in range(len(cell_plane_list)):
            cell_plane_tag = cell_plane_list[i]
            inner_plane_tag = inner_plane_list[i]
            # Create interface between 2 objects
            two_shapes, (outer_shape_map, inner_shape_map) = gmsh.model.occ.fragment(
                [(2, cell_plane_tag)], [(2, inner_plane_tag)]
            )
            gmsh.model.occ.synchronize()

            # Get the outer boundary
            outer_shell = gmsh.model.getBoundary(two_shapes, oriented=False)
            for i in range(len(outer_shell)):
                outer_marker_list.append(outer_shell[i][1])
            # Get the inner boundary
            inner_shell = gmsh.model.getBoundary(inner_shape_map, oriented=False)
            for i in range(len(inner_shell)):
                inner_marker_list.append(inner_shell[i][1])
            for tag in outer_shape_map:
                all_volumes.append(tag[1])
            for tag in inner_shape_map:
                inner_volume.append(tag[1])

            for vol in all_volumes:
                if vol not in inner_volume:
                    outer_volume.append(vol)

        # Add physical markers for facets
        if half_cell:  # if half, set symmetry axis to 0 (no flux)
            xmin, ymin, zmin = (-hInnerEdge / 10, -hInnerEdge / 10, -1)
            xmax, ymax, zmax = (hInnerEdge / 10, hInnerEdge / 10, max(zValsOuter) + 1)
            all_symm_bound = gmsh.model.occ.get_entities_in_bounding_box(
                xmin, ymin, zmin, xmax, ymax, zmax, dim=1
            )
            symm_bound_markers = []
            for i in range(len(all_symm_bound)):
                symm_bound_markers.append(all_symm_bound[i][1])
            # note that this first call sets the symmetry axis to tag 0 and
            # this is not overwritten by the next calls to add_physical_group
            gmsh.model.add_physical_group(1, symm_bound_markers, tag=0)
        gmsh.model.add_physical_group(1, outer_marker_list, tag=outer_marker)
        gmsh.model.add_physical_group(1, inner_marker_list, tag=interface_marker)

        # Physical markers for "volumes"
        gmsh.model.add_physical_group(2, outer_volume, tag=outer_tag)
        gmsh.model.add_physical_group(2, inner_volume, tag=inner_tag)

    def meshSizeCallback(dim, tag, x, y, z, lc):
        # mesh length is hEdge at the PM and hInnerEdge at the inner membrane
        # between these, the value is interpolated based on the relative distance
        # between the two membranes.
        # Inside the inner shape, the value is interpolated between hInnerEdge
        # and lc3, where lc3 = max(hInnerEdge, 0.2*maxInnerDim)
        # if innerRad=0, then the mesh length is interpolated between
        # hEdge at the PM and 0.2*maxOuterDim in the center
        rCur = np.sqrt(x**2 + y**2)
        RCur = np.sqrt(rCur**2 + (z - zMid) ** 2)
        outer_dist = np.sqrt((rCur - rValsOuter) ** 2 + (z - zValsOuter) ** 2)
        np.append(outer_dist, z)  # include the distance from the substrate
        dist_to_outer = min(outer_dist)
        if innerExpr == "":
            lc3 = 0.2 * maxOuterDim
            dist_to_inner = RCur
            in_outer = True
        else:
            inner_dist = np.sqrt((rCur - rValsInner) ** 2 + (z - zValsInner) ** 2)
            dist_to_inner = min(inner_dist)
            inner_idx = np.argmin(inner_dist)
            inner_rad = RInnerVec[inner_idx]
            R_rel_inner = RCur / inner_rad
            lc3 = max(hInnerEdge, 0.2 * maxInnerDim)
            in_outer = R_rel_inner > 1
        lc1 = hEdge
        lc2 = hInnerEdge
        if in_outer:
            lcTest = lc1 + (lc2 - lc1) * (dist_to_outer) / (dist_to_inner + dist_to_outer)
        else:
            lcTest = lc2 + (lc3 - lc2) * (1 - R_rel_inner)
        return lcTest

    gmsh.model.mesh.setSizeCallback(meshSizeCallback)
    # set off the other options for mesh size determination
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    # this changes the algorithm from Frontal-Delaunay to Delaunay,
    # which may provide better results when there are larger gradients in mesh size
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    gmsh.model.mesh.generate(2)
    rank = MPI.COMM_WORLD.rank
    tmp_folder = pathlib.Path(f"tmp_2DCell_{rank}")
    tmp_folder.mkdir(exist_ok=True)
    gmsh_file = tmp_folder / "2DCell.msh"
    gmsh.write(str(gmsh_file))
    gmsh.finalize()

    # return dolfin mesh of max dimension (parent mesh) and marker functions mf2 and mf3
    dmesh, mf2, mf3 = gmsh_to_dolfin(str(gmsh_file), tmp_folder, 2, comm)
    # remove tmp mesh and tmp folder
    gmsh_file.unlink(missing_ok=False)
    tmp_folder.rmdir()
    # return dolfin mesh, mf2 (2d tags) and mf3 (3d tags)
    if return_curvature:
        if innerExpr == "":
            facet_list = [outer_marker]
            cell_list = [outer_tag]
        else:
            facet_list = [outer_marker, interface_marker]
            cell_list = [outer_tag, inner_tag]
        if half_cell_with_curvature:  # will likely not work in parallel...
            dmesh_half, mf2_half, mf3_half = create_2Dcell(
                outerExpr,
                innerExpr,
                hEdge,
                hInnerEdge,
                interface_marker,
                outer_marker,
                inner_tag,
                outer_tag,
                comm,
                verbose,
                half_cell=True,
                return_curvature=False,
            )
            kappa_mf = compute_curvature(
                dmesh, mf2, mf3, facet_list, cell_list, half_mesh_data=(dmesh_half, mf2_half)
            )
            (dmesh, mf2, mf3) = (dmesh_half, mf2_half, mf3_half)
        else:
            kappa_mf = compute_curvature(dmesh, mf2, mf3, facet_list, cell_list)
        return (dmesh, mf2, mf3, kappa_mf)
    else:
        return (dmesh, mf2, mf3)


def create_2Dcell_xy(
    outerExpr: str = "",
    innerExpr: str = "",
    hEdge: float = 0,
    hInnerEdge: float = 0,
    interface_marker: int = 12,
    outer_marker: int = 10,
    inner_tag: int = 2,
    outer_tag: int = 1,
    comm: MPI.Comm = d.MPI.comm_world,
    verbose: bool = False,
    half_cell: bool = True,
    return_curvature: bool = False,
) -> Tuple[d.Mesh, d.MeshFunction, d.MeshFunction]:
    """
    Creates a 2D mesh of a cell profile, with the bounding curve defined in
    terms of r and z (e.g. unit circle would be "r**2 + (z-1)**2 - 1)
    It is assumed that substrate is present at z = 0, so if the curve extends
    below z = 0 , there is a sharp cutoff.
    If half_cell = True, only have of the contour is constructed, with a
    left zero-flux boundary at r = 0.
    Can include one compartment inside another compartment.
    Recommended for use with the axisymmetric feature of SMART.

    Args:
        outerExpr: String implicitly defining an r-z curve for the outer surface
        innerExpr: String implicitly defining an r-z curve for the inner surface
        hEdge: maximum mesh size at the outer edge
        hInnerEdge: maximum mesh size at the edge
            of the inner compartment
        interface_marker: The value to mark facets on the interface with
        outer_marker: The value to mark facets on edge of the outer ellipse with
        inner_tag: The value to mark the inner ellipse surface with
        outer_tag: The value to mark the outer ellipse surface with
        comm: MPI communicator to create the mesh with
        verbose: If true print gmsh output, else skip
        half_cell: If true, consider r=0 the symmetry axis for an axisymm shape
    Returns:
        Tuple (mesh, facet_marker, cell_marker)
    """
    import gmsh

    if outerExpr == "":
        ValueError("Outer surface is not defined")

    if return_curvature:
        # create full mesh for curvature analysis and then map onto half mesh
        # if half_cell_with_curvature is True
        half_cell_with_curvature = half_cell
        half_cell = False

    rValsOuter, zValsOuter = implicit_curve(outerExpr)

    if not innerExpr == "":
        rValsInner, zValsInner = implicit_curve(innerExpr)
        zMid = np.mean(zValsInner)
        ROuterVec = np.sqrt(rValsOuter**2 + (zValsOuter - zMid) ** 2)
        RInnerVec = np.sqrt(rValsInner**2 + (zValsInner - zMid) ** 2)
        maxOuterDim = max(ROuterVec)
        maxInnerDim = max(RInnerVec)
    else:
        zMid = np.mean(zValsOuter)
        ROuterVec = np.sqrt(rValsOuter**2 + (zValsOuter - zMid) ** 2)
        maxOuterDim = max(ROuterVec)
    if np.isclose(hEdge, 0):
        hEdge = 0.1 * maxOuterDim
    if np.isclose(hInnerEdge, 0):
        hInnerEdge = 0.2 * maxOuterDim if innerExpr == "" else 0.2 * maxInnerDim
    # Create the 2D mesh using gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", int(verbose))
    gmsh.model.add("2DCell")
    # first add outer body and revolve
    outer_tag_list = []
    for i in range(len(rValsOuter)):
        cur_tag = gmsh.model.occ.add_point(rValsOuter[i], zValsOuter[i], 0.0)
        outer_tag_list.append(cur_tag)
    outer_spline_tag = gmsh.model.occ.add_spline(outer_tag_list)
    if not half_cell:
        outer_tag_list2 = []
        for i in range(len(rValsOuter)):
            cur_tag = gmsh.model.occ.add_point(-rValsOuter[i], zValsOuter[i], 0.0)
            outer_tag_list2.append(cur_tag)
        outer_spline_tag2 = gmsh.model.occ.add_spline(outer_tag_list2)
    if np.isclose(zValsOuter[-1], 0):  # then include substrate at z=0
        if half_cell:
            origin_tag = gmsh.model.occ.add_point(0, 0, 0)
            symm_axis_tag = gmsh.model.occ.add_line(origin_tag, outer_tag_list[0])
            bottom_tag = gmsh.model.occ.add_line(origin_tag, outer_tag_list[-1])
            outer_loop_tag = gmsh.model.occ.add_curve_loop(
                [outer_spline_tag, bottom_tag, symm_axis_tag]
            )
        else:
            bottom_tag = gmsh.model.occ.add_line(outer_tag_list[-1], outer_tag_list2[-1])
            outer_loop_tag = gmsh.model.occ.add_curve_loop(
                [outer_spline_tag, outer_spline_tag2, bottom_tag]
            )
    else:
        if half_cell:
            symm_axis_tag = gmsh.model.occ.add_line(outer_tag_list[0], outer_tag_list[-1])
            outer_loop_tag = gmsh.model.occ.add_curve_loop([outer_spline_tag, symm_axis_tag])
        else:
            outer_loop_tag = gmsh.model.occ.add_curve_loop([outer_spline_tag, outer_spline_tag2])
    cell_plane_tag = gmsh.model.occ.add_plane_surface([outer_loop_tag])

    if innerExpr == "":
        # No inner shape in this case
        gmsh.model.occ.synchronize()
        gmsh.model.add_physical_group(2, [cell_plane_tag], tag=outer_tag)
        facets = gmsh.model.getBoundary([(2, cell_plane_tag)])
        facet_tag_list = []
        for i in range(len(facets)):
            facet_tag_list.append(facets[i][1])
        if half_cell:  # if half, set symmetry axis to 0 (no flux)
            xmin, ymin, zmin = (-hInnerEdge / 10, -1, -hInnerEdge / 10)
            xmax, ymax, zmax = (hInnerEdge / 10, max(zValsOuter) + 1, hInnerEdge / 10)
            all_symm_bound = gmsh.model.occ.get_entities_in_bounding_box(
                xmin, ymin, zmin, xmax, ymax, zmax, dim=1
            )
            symm_bound_markers = []
            for i in range(len(all_symm_bound)):
                symm_bound_markers.append(all_symm_bound[i][1])
            gmsh.model.add_physical_group(1, symm_bound_markers, tag=0)
        gmsh.model.add_physical_group(1, facet_tag_list, tag=outer_marker)
    else:
        # Add inner shape
        inner_tag_list = []
        for i in range(len(rValsInner)):
            cur_tag = gmsh.model.occ.add_point(rValsInner[i], zValsInner[i], 0.0)
            inner_tag_list.append(cur_tag)
        inner_spline_tag = gmsh.model.occ.add_spline(inner_tag_list)
        if half_cell:
            symm_inner_tag = gmsh.model.occ.add_line(inner_tag_list[0], inner_tag_list[-1])
            inner_loop_tag = gmsh.model.occ.add_curve_loop([inner_spline_tag, symm_inner_tag])
        else:
            inner_tag_list2 = []
            for i in range(len(rValsInner)):
                cur_tag = gmsh.model.occ.add_point(-rValsInner[i], zValsInner[i], 0.0)
                inner_tag_list2.append(cur_tag)
            inner_spline_tag2 = gmsh.model.occ.add_spline(inner_tag_list2)
            inner_loop_tag = gmsh.model.occ.add_curve_loop([inner_spline_tag, inner_spline_tag2])
        inner_plane_tag = gmsh.model.occ.add_plane_surface([inner_loop_tag])
        cell_plane_list = [cell_plane_tag]
        inner_plane_list = [inner_plane_tag]

        outer_volume = []
        inner_volume = []
        all_volumes = []
        inner_marker_list = []
        outer_marker_list = []
        for i in range(len(cell_plane_list)):
            cell_plane_tag = cell_plane_list[i]
            inner_plane_tag = inner_plane_list[i]
            # Create interface between 2 objects
            two_shapes, (outer_shape_map, inner_shape_map) = gmsh.model.occ.fragment(
                [(2, cell_plane_tag)], [(2, inner_plane_tag)]
            )
            gmsh.model.occ.synchronize()

            # Get the outer boundary
            outer_shell = gmsh.model.getBoundary(two_shapes, oriented=False)
            for i in range(len(outer_shell)):
                outer_marker_list.append(outer_shell[i][1])
            # Get the inner boundary
            inner_shell = gmsh.model.getBoundary(inner_shape_map, oriented=False)
            for i in range(len(inner_shell)):
                inner_marker_list.append(inner_shell[i][1])
            for tag in outer_shape_map:
                all_volumes.append(tag[1])
            for tag in inner_shape_map:
                inner_volume.append(tag[1])

            for vol in all_volumes:
                if vol not in inner_volume:
                    outer_volume.append(vol)

        # Add physical markers for facets
        if half_cell:  # if half, set symmetry axis to 0 (no flux)
            xmin, ymin, zmin = (-hInnerEdge / 10, -1, -hInnerEdge / 10)
            xmax, ymax, zmax = (hInnerEdge / 10, max(zValsOuter) + 1, hInnerEdge / 10)
            all_symm_bound = gmsh.model.occ.get_entities_in_bounding_box(
                xmin, ymin, zmin, xmax, ymax, zmax, dim=1
            )
            symm_bound_markers = []
            for i in range(len(all_symm_bound)):
                symm_bound_markers.append(all_symm_bound[i][1])
            # note that this first call sets the symmetry axis to tag 0 and
            # this is not overwritten by the next calls to add_physical_group
            gmsh.model.add_physical_group(1, symm_bound_markers, tag=0)
        gmsh.model.add_physical_group(1, outer_marker_list, tag=outer_marker)
        gmsh.model.add_physical_group(1, inner_marker_list, tag=interface_marker)

        # Physical markers for "volumes"
        gmsh.model.add_physical_group(2, outer_volume, tag=outer_tag)
        gmsh.model.add_physical_group(2, inner_volume, tag=inner_tag)

    def meshSizeCallback(dim, tag, x, y, z, lc):
        # mesh length is hEdge at the PM and hInnerEdge at the inner membrane
        # between these, the value is interpolated based on the relative distance
        # between the two membranes.
        # Inside the inner shape, the value is interpolated between hInnerEdge
        # and lc3, where lc3 = max(hInnerEdge, 0.2*maxInnerDim)
        # if innerRad=0, then the mesh length is interpolated between
        # hEdge at the PM and 0.2*maxOuterDim in the center
        rCur = np.sqrt(x**2 + z**2)
        RCur = np.sqrt(rCur**2 + (y - zMid) ** 2)
        outer_dist = np.sqrt((rCur - rValsOuter) ** 2 + (z - zValsOuter) ** 2)
        np.append(outer_dist, z)  # include the distance from the substrate
        dist_to_outer = min(outer_dist)
        if innerExpr == "":
            lc3 = 0.2 * maxOuterDim
            dist_to_inner = RCur
            in_outer = True
        else:
            inner_dist = np.sqrt((rCur - rValsInner) ** 2 + (y - zValsInner) ** 2)
            dist_to_inner = min(inner_dist)
            inner_idx = np.argmin(inner_dist)
            inner_rad = RInnerVec[inner_idx]
            R_rel_inner = RCur / inner_rad
            lc3 = max(hInnerEdge, 0.2 * maxInnerDim)
            in_outer = R_rel_inner > 1
        lc1 = hEdge
        lc2 = hInnerEdge
        if in_outer:
            lcTest = lc1 + (lc2 - lc1) * (dist_to_outer) / (dist_to_inner + dist_to_outer)
        else:
            lcTest = lc2 + (lc3 - lc2) * (1 - R_rel_inner)
        return lcTest

    gmsh.model.mesh.setSizeCallback(meshSizeCallback)
    # set off the other options for mesh size determination
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    # this changes the algorithm from Frontal-Delaunay to Delaunay,
    # which may provide better results when there are larger gradients in mesh size
    gmsh.option.setNumber("Mesh.Algorithm", 5)

    gmsh.model.mesh.generate(2)
    rank = MPI.COMM_WORLD.rank
    tmp_folder = pathlib.Path(f"tmp_2DCell_{rank}")
    tmp_folder.mkdir(exist_ok=True)
    gmsh_file = tmp_folder / "2DCell.msh"
    gmsh.write(str(gmsh_file))
    gmsh.finalize()

    # return dolfin mesh of max dimension (parent mesh) and marker functions mf2 and mf3
    dmesh, mf1, mf2 = gmsh_to_dolfin(str(gmsh_file), tmp_folder, 2, comm)
    # remove tmp mesh and tmp folder
    gmsh_file.unlink(missing_ok=False)
    tmp_folder.rmdir()
    # return dolfin mesh, mf1 (1d tags) and mf2 (2d tags)
    if return_curvature:
        if innerExpr == "":
            facet_list = [outer_marker]
            cell_list = [outer_tag]
        else:
            facet_list = [outer_marker, interface_marker]
            cell_list = [outer_tag, inner_tag]
        if half_cell_with_curvature:  # will likely not work in parallel...
            dmesh_half, mf1_half, mf2_half = create_2Dcell(
                outerExpr,
                innerExpr,
                hEdge,
                hInnerEdge,
                interface_marker,
                outer_marker,
                inner_tag,
                outer_tag,
                comm,
                verbose,
                half_cell=True,
                return_curvature=False,
            )
            kappa_mf = compute_curvature(
                dmesh, mf1, mf2, facet_list, cell_list, half_mesh_data=(dmesh_half, mf1_half)
            )
            (dmesh, mf1, mf2) = (dmesh_half, mf1_half, mf2_half)
        else:
            kappa_mf = compute_curvature(dmesh, mf1, mf2, facet_list, cell_list)
        return (dmesh, mf1, mf2, kappa_mf)
    else:
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
    Args:
        gmsh_file_name: .msh file (string)
        tmp_folder_name: folder name to store temporary mesh files
        dimension: dimension of parent mesh (int - either 2 or 3)
    Returns:
        Tuple containing:
            dMesh: Dolfin-style parent mesh
            mf_facet: markers for facets
            mf_cell: markers for cells
    """
    import meshio

    tmp_file_cell = tmp_folder / "tempmesh_cell.xdmf"
    tmp_file_facet = tmp_folder / "tempmesh_facet.xdmf"

    if comm.rank == 0:
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
        if dimension == 2 and np.all(mesh_in.points[:, 2] == 0):
            mesh_in.points = mesh_in.points[:, :2]  # then prune z values
        out_mesh_cell = meshio.Mesh(
            points=mesh_in.points,
            cells={cell_type: cells},
            cell_data={"mf_data": [cell_data]},
        )
        meshio.write(tmp_file_cell, out_mesh_cell)
        # convert facet mesh
        facets = mesh_in.get_cells_type(facet_type)
        facet_data = mesh_in.get_cell_data("gmsh:physical", facet_type)  # extract values of tags
        out_mesh_facet = meshio.Mesh(
            points=mesh_in.points,
            cells={facet_type: facets},
            cell_data={"mf_data": [facet_data]},
        )
        meshio.write(tmp_file_facet, out_mesh_facet)

    comm.Barrier()
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
    if comm.rank == 0:
        tmp_file_cell.unlink(missing_ok=False)
        tmp_file_cell.with_suffix(".h5").unlink(missing_ok=False)
        tmp_file_facet.unlink(missing_ok=False)
        tmp_file_facet.with_suffix(".h5").unlink(missing_ok=False)
    # return dolfin mesh and mfs (marker functions)
    return (dmesh, mf_facet, mf_cell)


def write_mesh(
    mesh: d.Mesh,
    mf_facet: d.MeshFunction,
    mf_cell: d.MeshFunction,
    filename: pathlib.Path = pathlib.Path("DemoMesh.h5"),
    subdomains=None,
):
    """
    Write 3D mesh, with cell markers (mf_cell)
    and facet markers (mf_facet) to hdf5 file and pvd files.
    """
    if isinstance(filename, str):
        filename = pathlib.Path(filename)
    comm = mesh.mpi_comm()
    # extract dimensionality for meshfunctions
    cell_dim = mf_cell.dim()
    facet_dim = mf_facet.dim()
    # Write mesh and meshfunctions to file
    with d.HDF5File(comm, str(filename.with_suffix(".h5")), "w") as hdf5:
        hdf5.write(mesh, "/mesh")
        hdf5.write(mf_cell, f"/mf{cell_dim}")
        hdf5.write(mf_facet, f"/mf{facet_dim}")
        if subdomains is not None:
            for i in range(len(subdomains)):
                cur_dim = subdomains[i].dim()
                hdf5.write(subdomains[i], f"/subdomain{i}_{cur_dim}")


class LoadedMesh(NamedTuple):
    mesh: d.Mesh
    mf_cell: d.MeshFunction
    mf_facet: d.MeshFunction
    subdomains: list
    filename: pathlib.Path


def load_mesh(
    filename: Union[pathlib.Path, str],
    comm: MPI.Intracomm = MPI.COMM_WORLD,
    mesh: Optional[d.Mesh] = None,
    extra_keys: Optional[list[str]] = None,
) -> LoadedMesh:
    """Load mesh and mesh functions from file


    Args:
        filename : Path to the file with the mesh
        comm : MPI communicator, by default MPI.COMM_WORLD
        mesh : The mesh, by default None. If provided, only the meshfunctions
            will be loaded from the file assuming that the given mesh is the
            mesh used for the mesh functions

    Returns: A named tuple with the mesh, cell function, facet function
        and the filename

    """
    if extra_keys is None:
        extra_keys = []

    if isinstance(filename, str):
        filename = pathlib.Path(filename)

    if not filename.is_file():
        raise FileNotFoundError(f"File {filename} does not exists")

    if mesh is None:
        mesh = d.Mesh(comm)

        with d.HDF5File(comm, filename.with_suffix(".h5").as_posix(), "r") as hdf5:
            hdf5.read(mesh, "/mesh", False)

    dim = mesh.topology().dim()  # geometric_dimension()
    mf_cell = d.MeshFunction("size_t", mesh, dim)
    mf_facet = d.MeshFunction("size_t", mesh, dim - 1)
    subdomains = []

    with d.HDF5File(mesh.mpi_comm(), str(filename.with_suffix(".h5")), "r") as hdf5:
        hdf5.read(mf_cell, f"/mf{dim}")
        hdf5.read(mf_facet, f"/mf{dim-1}")
        for key in extra_keys:
            cur_dim = int(key[-1])
            mf_cur = d.MeshFunction("size_t", mesh, cur_dim)
            hdf5.read(mf_cur, f"/{key}")
            subdomains.append(mf_cur)

    return LoadedMesh(
        mesh=mesh, mf_cell=mf_cell, mf_facet=mf_facet, filename=filename, subdomains=subdomains
    )
