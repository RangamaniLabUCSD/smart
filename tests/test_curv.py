from smart import mesh_tools
import numpy as np


def test_circle_curv():
    """Test calculation of curvature for unit circle"""
    cell_marker = 1
    bound_marker = 10
    circle_mesh, mf1_circle, mf2_circle = mesh_tools.create_ellipses(
        1.0, 1.0, hEdge=0.1, outer_tag=cell_marker, outer_marker=bound_marker
    )
    curv_circle = mesh_tools.compute_curvature(
        circle_mesh, mf1_circle, mf2_circle, [bound_marker], [cell_marker]
    )
    curv_circle_vec = curv_circle.array()
    curv_circle_vec = curv_circle_vec[curv_circle_vec != 0]
    circle_error = 100 * np.abs(curv_circle_vec - 1)
    print(
        f"Maximum error in circle curvature is {max(circle_error)}%"
        f" and mean value is {np.average(curv_circle_vec)}"
    )


def test_sphere_curv():
    """Test calculation of curvature for unit sphere"""
    cell_marker = 1
    bound_marker = 10
    sphere_mesh, mf2_sphere, mf3_sphere = mesh_tools.create_spheres(
        outerRad=1.0, hEdge=0.1, outer_vol_tag=cell_marker, outer_marker=bound_marker
    )
    curv_sphere = mesh_tools.compute_curvature(
        sphere_mesh, mf2_sphere, mf3_sphere, [bound_marker], [cell_marker]
    )
    curv_sphere_vec = curv_sphere.array()
    curv_sphere_vec = curv_sphere_vec[curv_sphere_vec != 0]
    sphere_error = 100 * np.abs(curv_sphere_vec - 1)
    print(
        f"Maximum error in sphere curvature is {max(sphere_error)}%"
        f" and mean value is {np.average(curv_sphere_vec)}"
    )


def test_axisymm_curv():
    """
    Test calculation of curvature for unit sphere defined
    as a half-circle rotated about a central axis
    """
    cell_marker = 1
    bound_marker = 10
    half_circle_mesh, mf1, mf2, curv_half = mesh_tools.create_2Dcell(
        outerExpr="r**2 + (z-2)**2 - 1",
        hEdge=0.1,
        half_cell=True,
        return_curvature=True,
        outer_tag=cell_marker,
        outer_marker=bound_marker,
    )
    curv_half_vec = curv_half.array()
    curv_half_vec = curv_half_vec[curv_half_vec != 0]
    half_error = 100 * np.abs(curv_half_vec - 1)
    print(
        f"Maximum error in axisymmetric curvature is {max(half_error)}%",
        f" and mean value is {np.average(curv_half_vec)}",
    )
