"""
General functions: array manipulation, data i/o, etc
"""
import os
import time
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple, Union

import dolfin as d
import gmsh
import meshio

# import trimesh
import numpy as np
import pint

import ufl
from pytz import timezone
from termcolor import colored

from .deprecation import deprecated
from .config import global_settings as gset, FancyFormatter
from .units import unit

__all__ = [
    "stubs_expressions",
    "sub",
    "pint_unit_to_quantity",
    "pint_quantity_to_unit",
    "Stopwatch",
    "facet_topology",
    "cube_condition",
    "DemoCuboidsMesh",
    "DemoSpheresMesh",
    "write_mesh",
]


comm = d.MPI.comm_world
rank = comm.rank
size = comm.size
root = 0

logger = logging.getLogger(__name__)


def stubs_expressions(
    dolfin_expressions: Dict[str, Callable[[Any], Any]]
) -> Dict[str, Callable[[Any], Union[ufl.core.expr.Expr, float]]]:
    """
    Map strings to DOLFIN/UFL functions, that takes in
    `stubs`-Expressions, i.e. functions with a unit.

    Args:
        dolfin_expressions: Dictonary of strings mapping
          to stubs expressions

    Example:

        .. highlight:: python
        .. code-block:: python

            input = {"sin": ufl.sin}
            output = stubs_expressions(input)

        Output is then a dictionary that maps "sin" to
        a function :code:`sin(x)` that takes in a
        function with a unit and returns
        :code:`ufl.sin(x.to(unit.dimensionless).magnitude)`
    """
    return {
        k: lambda x: v(x.to(unit.dimensionless).magnitude) for k, v in dolfin_expressions.items()
    }


def sub(
    func: Union[
        List[Union[d.Function, d.FunctionSpace]],
        d.Function,
        d.MixedFunctionSpace,
        d.FunctionSpace,
        d.function.argument.Argument,
    ],
    idx: int,
    collapse_function_space: bool = True,
):
    """
    A utility function for getting the sub component of a

    - Functionspace
    - Mixed function space
    - Function
    - Test/Trial Function

    Args:
        func: The input
        idx: The component to extract
        collapse_function_space: If the input `func`
        is a `dolfin.FunctionSpace`, collapse it if `True`.

    Returns:
        The relevant component of the input
    """

    if isinstance(func, (list, tuple)):
        return func[idx]

    elif isinstance(func, d.Function):
        # MixedFunctionSpace
        if func._functions:
            assert func.sub(idx) is func._functions[idx]
            return func.sub(idx)
        else:
            # scalar FunctionSpace
            if func.num_sub_spaces() <= 1 and idx == 0:
                return func
            else:
                # Use d.split() to get subfunctions from a
                # VectorFunctionSpace that can be used in forms
                return func.sub(idx)

    elif isinstance(func, d.MixedFunctionSpace):
        return func.sub(idx)

    elif isinstance(func, d.FunctionSpace):
        if func.num_sub_spaces() <= 1 and idx == 0:
            return func
        else:
            if collapse_function_space:
                return func.sub(idx).collapse()
            else:
                return func.sub(idx)

    elif isinstance(func, d.function.argument.Argument):
        if func.function_space().num_sub_spaces() <= 1 and idx == 0:
            return func
        else:
            return func[idx]

    else:
        raise ValueError(f"Unknown input type of {func=}")


def pint_unit_to_quantity(pint_unit):
    if not isinstance(pint_unit, pint.Unit):
        raise TypeError("Input must be a pint unit")
    # returning pint.Quantity(1, pint_unit) changes the unit
    # registry which we do NOT want
    return 1.0 * pint_unit


def pint_quantity_to_unit(pint_quantity):
    if not isinstance(pint_quantity, pint.Quantity):
        raise TypeError("Input must be a pint quantity")
    if pint_quantity.magnitude != 1.0:
        raise ValueError("Trying to convert a pint quantity into a unit with magnitude != 1")
    return pint_quantity.units


# Write a stopwatch class to measure time elapsed
# with a start, stop, and pause methods
# Keep track of timings in a list of lists called
# self.timings, each time the timer is paused,
# the time elapsed since the last pause is added to
# the sublist. Using stop resets the timer to zero
# and beings a new list of timings.


class Stopwatch:
    "Basic stopwatch class with inner/outer timings (pause and stop)"

    def __init__(self, name=None, time_unit="s", print_buffer=0, filename=None, start=False):
        self.name = name
        self.time_unit = time_unit
        self.stop_timings = []  # length = number of stops
        self.pause_timings = []  # length = number of stops (list of lists)
        self._pause_timings = []  # length = number of pauses (reset on stop)
        self._times = []
        self.is_paused = True
        self.print_buffer = print_buffer
        self._print_name = f"{str(self.name): <{self.print_buffer}}"
        # self.start()
        self.filename = filename
        if filename is not None:
            self.file_logger = logging.getLogger("stubs_stop_watch")
            handler = logging.FileHandler(filename=filename)
            handler.setFormatter(FancyFormatter())
            self.file_logger.addHandler(handler)
            self.file_logger.setLevel(logging.DEBUG)
        else:
            # Just use the regular logger
            self.file_logger = logger
        if start:
            self.start()

    def start(self):
        self._times.append(time.time())
        self.is_paused = False

    def pause(self):
        if self.is_paused:
            return
        else:
            self._times.append(time.time())
            self._pause_timings.append(self._times[-1] - self._times[-2])
            self.is_paused = True
            self.file_logger.debug(
                f"{self.name} (iter {len(self._pause_timings)}) finished "
                f"in {self.time_str(self._pause_timings[-1])} {self.time_unit}",
                extra=dict(format_type="logred"),
            )

    def stop(self, print_result=True):
        self._times.append(time.time())
        if self.is_paused:
            final_time = 0
        else:
            final_time = self._times[-1] - self._times[-2]
            self.is_paused = True
        total_time = sum(self._pause_timings) + final_time
        self.stop_timings.append(total_time)
        if print_result:
            self.file_logger.debug(
                f"{self._print_name} finished in {self.time_str(total_time)} {self.time_unit}",
                extra=dict(format_type="logred"),
            )

        self.pause_timings.append(self._pause_timings)
        self._pause_timings = []
        self._times = []

    def set_timing(self, timing):
        self.stop_timings.append(timing)
        self.file_logger.debug(
            f"{self._print_name} finished in {self.time_str(timing)} {self.time_unit}",
            extra=dict(format_type="logred"),
        )

    def print_last_stop(self):
        self.file_logger.debug(
            f"{self._print_name} finished in "
            f"{self.time_str(self.stop_timings[-1])} {self.time_unit}",
            extra=dict(format_type="logred"),
        )

    def time_str(self, t):
        return str({"us": 1e6, "ms": 1e3, "s": 1, "min": 1 / 60}[self.time_unit] * t)[0:8]


@deprecated
def _fancy_print(
    title_text,
    buffer_color=None,
    text_color=None,
    filler_char=None,
    num_banners=None,
    new_lines=None,
    left_justify=None,
    format_type="default",
    include_timestamp=True,
    filename=None,
):
    "Formatted text to stand out."

    # Initialize with the default options
    buffer_color_ = "cyan"
    text_color_ = "green"
    filler_char_ = "="
    num_banners_ = 0
    new_lines_ = [0, 0]
    left_justify_ = False
    # Override with format_type options
    if format_type == "default":
        pass
    elif format_type == "title":
        text_color_ = "magenta"
        num_banners_ = 1
        new_lines_ = [1, 0]
    elif format_type == "subtitle":
        text_color_ = "green"
        filler_char_ = "."
        left_justify_ = True
    elif format_type == "data":
        buffer_color_ = "white"
        text_color_ = "white"
        filler_char_ = ""
        left_justify_ = True
    elif format_type == "data_important":
        buffer_color_ = "white"
        text_color_ = "red"
        filler_char_ = ""
        left_justify_ = True
    elif format_type == "log":
        buffer_color_ = "white"
        text_color_ = "green"
        filler_char_ = ""
        left_justify_ = True
    elif format_type == "logred":
        buffer_color_ = "white"
        text_color_ = "red"
        filler_char_ = ""
        left_justify_ = True
    elif format_type == "log_important":
        buffer_color_ = "white"
        text_color_ = "magenta"
        filler_char_ = "."
    elif format_type == "log_urgent":
        buffer_color_ = "white"
        text_color_ = "red"
        filler_char_ = "."
    elif format_type == "warning":
        buffer_color_ = "magenta"
        text_color_ = "red"
        filler_char_ = "!"
        num_banners_ = 2
        new_lines_ = [1, 1]
    elif format_type == "timestep":
        text_color_ = "red"
        num_banners_ = 2
        filler_char_ = "."
        new_lines_ = [1, 1]
    elif format_type == "solverstep":
        text_color_ = "red"
        num_banners_ = 1
        filler_char_ = "."
        new_lines_ = [1, 1]
    elif format_type == "assembly":
        text_color_ = "magenta"
        num_banners_ = 0
        filler_char_ = "."
        new_lines_ = [1, 0]
    elif format_type == "assembly_sub":
        text_color_ = "magenta"
        num_banners_ = 0
        filler_char_ = ""
        new_lines_ = [0, 0]
        left_justify_ = True
    elif format_type is not None:
        raise ValueError("Unknown formatting_type.")

    # Override again with user options
    if buffer_color is None:
        buffer_color = buffer_color_
    if text_color is None:
        text_color = text_color_
    if filler_char is None:
        filler_char = filler_char_
    if num_banners is None:
        num_banners = num_banners_
    if new_lines is None:
        new_lines = new_lines_
    if left_justify is None:
        left_justify = left_justify_

    # include MPI rank in message
    if size > 1:
        title_text = f"CPU {rank}: {title_text}"
    if include_timestamp:
        timestamp = datetime.now(timezone("US/Pacific")).strftime("[%Y-%m-%d time=%H:%M:%S]")
        title_text = f"{timestamp} {title_text}"

    # calculate optimal buffer size
    min_buffer_size = 5
    terminal_width = 120
    buffer_size = max(
        [min_buffer_size, int((terminal_width - 1 - len(title_text)) / 2 - 1)]
    )  # terminal width == 80
    title_str_len = (buffer_size + 1) * 2 + len(title_text)
    parity = 1 if title_str_len == 78 else 0

    # color/stylize buffer, text, and banner
    def buffer(buffer_size):
        return colored(filler_char * buffer_size, buffer_color)

    if left_justify:
        title_str = f"{colored(title_text, text_color)} {buffer(buffer_size*2+1+parity)}"
    else:
        title_str = (
            f"{buffer(buffer_size)} {colored(title_text, text_color)} "
            f"{buffer(buffer_size+parity)}"
        )
    banner = colored(filler_char * (title_str_len + parity), buffer_color)

    def print_out(text, filename=None):
        "print to file and terminal"
        if filename is not None:
            with open(filename, "a") as f:
                f.write(text + "\n")
        elif gset["log_filename"] is not None:
            with open(gset["log_filename"], "a") as f:
                f.write(text + "\n")
        print(text)

    # initial spacing
    if new_lines[0] > 0:
        print_out("\n" * (new_lines[0] - 1), filename)
    # print first banner
    for _ in range(num_banners):
        print_out(f"{banner}", filename)
    # print main text
    print_out(title_str, filename)
    # print second banner
    for _ in range(num_banners):
        print_out(f"{banner}", filename)
    # end spacing
    if new_lines[1] > 0:
        print_out("\n" * (new_lines[1] - 1), filename)


# Get cells and faces for each subdomain


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
    Returns:
        A triplet (mesh, facet_marker, cell_marker)
    """
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
    gmsh.write("twoSpheres.msh")  # save locally
    gmsh.finalize()

    # load, convert to xdmf, and save as temp files
    mesh3d_in = meshio.read("twoSpheres.msh")

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
    meshio.write("tempmesh_3dout.xdmf", tet_mesh)
    meshio.write("tempmesh_2dout.xdmf", tri_mesh)

    # convert xdmf mesh to dolfin-style mesh
    dmesh = d.Mesh()
    mvc3 = d.MeshValueCollection("size_t", dmesh, 3)
    with d.XDMFFile("tempmesh_3dout.xdmf") as infile:
        infile.read(dmesh)
        infile.read(mvc3, "mf_data")
    mf3 = d.cpp.mesh.MeshFunctionSizet(dmesh, mvc3)
    # set unassigned volumes to tag=0
    mf3.array()[np.where(mf3.array() > 1e9)[0]] = 0
    mvc2 = d.MeshValueCollection("size_t", dmesh, 2)
    with d.XDMFFile("tempmesh_2dout.xdmf") as infile:
        infile.read(mvc2, "mf_data")
    mf2 = d.cpp.mesh.MeshFunctionSizet(dmesh, mvc2)
    # set inner faces to tag=0
    mf2.array()[np.where(mf2.array() > 1e9)[0]] = 0

    # use os to remove temp meshes
    os.remove("tempmesh_2dout.xdmf")
    os.remove("tempmesh_3dout.xdmf")
    os.remove("tempmesh_2dout.h5")
    os.remove("tempmesh_3dout.h5")
    os.remove("twoSpheres.msh")
    # return dolfin mesh, mf2 (2d tags) and mf3 (3d tags)
    return (dmesh, mf2, mf3)


def write_mesh(mesh, mf2, mf3, filename="DemoCuboidMesh"):
    # Write mesh and meshfunctions to file
    hdf5 = d.HDF5File(mesh.mpi_comm(), filename + ".h5", "w")
    hdf5.write(mesh, "/mesh")
    hdf5.write(mf3, "/mf3")
    hdf5.write(mf2, "/mf2")
    # For visualization of domains
    d.File(f"{filename}_mf3.pvd") << mf3
    d.File(f"{filename}_mf2.pvd") << mf2
