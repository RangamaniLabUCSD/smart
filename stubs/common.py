"""
General functions: array manipulation, data i/o, etc
"""
import functools
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import dolfin as d
# import trimesh
import numpy as np
import pandas
import pint
import scipy.interpolate as interp
import sympy
import ufl
from pytz import timezone
from termcolor import colored

import stubs

from .config import global_settings as gset

from .units import unit

__all__ = [
    "deprecated",
    "nan_to_none"
    "stubs_expression",
    "sub",
    "ref",
    "insert_dataframe_col",
    "submesh_dof_to_mesh_dof",
    "submesh_dof_to_vertex",
    "submesh_to_bmesh",
    "bmesh_to_parent",
    "mesh_vertex_to_dof",
    "round_to_n",
    "interp_limit_dy",
    "sum_discrete_signals",
    "np_smart_hstack",
    "append_meshfunction_to_meshdomains",
    "pint_unit_to_quantity",
    "pint_quantity_to_unit",
    "convert_xml_to_hdf5",
    "read_hdf5",
    "data_path",
    "Stopwatch",
    "find_steady_state",
]

comm = d.MPI.comm_world
rank = comm.rank
size = comm.size
root = 0


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        warnings.warn(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            stacklevel=2,
        )
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def stubs_expressions(
    dolfin_expressions: Dict[str, Callable[[Any], Any]]
) -> Dict[str, Callable[[Any], Union[ufl.core.expr.Expr, float]]]:
    """
    Map strings to DOLFIN/UFL functions, that takes in `stubs`-Expressions, i.e. functions with a unit.

    Example:

        .. highlight:: python
        .. code-block:: python

            input = {"sin": ufl.sin}
            output = stubs_expressions(input)

        Output is then a dictionary that maps "sin" to a function `sin(x)` that takes in a function with a unit and
        returns `ufl.sin(x.to(unit.dimensionless).magnitude)`
    """
    return {
        k: lambda x: v(x.to(unit.dimensionless).magnitude)
        for k, v in dolfin_expressions.items()
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
        collapse_function_space: If the input `func` is a `dolfin.FunctionSpace`, collapse it if `True`.
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
                # Use d.split() to get subfunctions from a VectorFunctionSpace that can be used in forms
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


@deprecated
class ref:
    """
    Pandas dataframe doesn't play nice with dolfin indexed functions since it will try
    to take the length but dolfin will not return anything. We use ref() to trick pandas
    into treating the dolfin indexed function as a normal object rather than something
    with an associated length
    """

    def __init__(self, obj):
        self.obj = obj

    def get(self):
        return self.obj

    def set(self, obj):
        self.obj = obj


@deprecated
def insert_dataframe_col(
    df: pandas.DataFrame,
    columnName: str,
    columnNumber: Optional[int] = None,
    items=None,
):
    """
    pandas requires some weird manipulation to insert some things, e.g. dictionaries, as dataframe entries
    items is a list of things
    by default, will insert a column of empty dicts as the last column
    """
    if items is None:
        items = [{} for x in range(df.shape[0])]
    if columnNumber == None:
        columnNumber = df.columns.size
    df.insert(columnNumber, columnName, items)


@deprecated
def submesh_dof_to_mesh_dof(
    Vsubmesh,
    submesh,
    bmesh_emap_0,
    V,
    submesh_species_index=0,
    mesh_species_index=0,
    index=None,
):
    """
    Takes dof indices (single index or a list) on a submesh of a boundarymesh of a mesh and returns
    the dof indices of the original mesh.
    Based of the following forum posts:
    https://fenicsproject.org/qa/13595/interpret-vertex_to_dof_map-dof_to_vertex_map-function/
    https://fenicsproject.org/qa/6810/vertex-mapping-from-submesh-boundarymesh-back-actual-mesh/
    """
    idx = submesh_dof_to_vertex(Vsubmesh, submesh_species_index, index)
    idx = submesh_to_bmesh(submesh, idx)
    idx = bmesh_to_parent(bmesh_emap_0, idx)
    idx = mesh_vertex_to_dof(V, mesh_species_index, idx)
    return idx


@deprecated
def submesh_dof_to_vertex(Vsubmesh: d.FunctionSpace, species_index: int, index: Optional[List[int]] = None):
    """
    Map a Lagrange-1 function on a sub-mesh to the vertices of the mesh
    DOFLIN DOF indices are not guaranteed to be equivalent to the vertex indices of the corresponding mesh
    """
    num_spaces = Vsubmesh.num_sub_spaces()
    num_species = max(num_spaces, 1)

    # This assumption of the number of dofs per sub spaces/species is fine as `dof_to_vertex_map` only
    # supports first order Lagrange functions.
    num_dofs_local = int(len(Vsubmesh.dofmap().dofs()) / num_species)
    if index is None:
        index = range(num_dofs_local)

    dof_to_vertex = d.dof_to_vertex_map(Vsubmesh)
    mapping = dof_to_vertex[range(species_index, len(dof_to_vertex), num_species)] / num_species
    mapping = [int(x) for x in mapping]
    return [mapping[x] for x in index]


@deprecated
def submesh_to_bmesh(submesh, index):
    submesh_to_bmesh_vertex = submesh.data().array("parent_vertex_indices", 0)
    return submesh_to_bmesh_vertex[index]


@deprecated
def bmesh_to_parent(bmesh_emap_0, index):
    if (
        bmesh_emap_0.max() > 1e9
    ):  # unless the mesh has 1e9 vertices this is a sign of an error
        raise Exception("Error in bmesh_emap.")
    return bmesh_emap_0[index]


@deprecated
def mesh_vertex_to_dof(V, species_index, index):
    num_species = V.num_sub_spaces()
    if num_species == 0:
        num_species = 1

    mapping = d.vertex_to_dof_map(V)

    mapping = mapping[range(species_index, len(mapping), num_species)]

    return [mapping[x] for x in index]


def round_to_n(x: np.float64, n: int) -> int:
    """
    Rounds to n sig figs
    """
    if x == 0:
        return 0
    else:
        sign = np.sign(x)
        x = np.abs(x)
        return sign * round(x, -int(np.floor(np.log10(x))) + (n - 1))


@deprecated
def interp_limit_dy(t, y, max_dy, interp_type: str = "linear"):
    """
    Interpolates t and y such that dy between time points is never greater than max_dy
    Maintains all original t and y
    """
    interp_t = t.reshape(-1, 1)
    dy_vec = y[1:] - y[:-1]

    for idx, dy in enumerate(dy_vec):
        npoints = np.int(np.ceil(np.abs(dy / max_dy))) - 1
        if npoints >= 1:
            new_t = np.linspace(t[idx], t[idx + 1], npoints + 2)[1:-1]
            interp_t = np.vstack([interp_t, new_t.reshape(-1, 1)])

    interp_t = np.sort(
        interp_t.reshape(
            -1,
        )
    )
    interp_y = interp.interp1d(t, y, kind=interp_type)(interp_t)

    return (interp_t, interp_y)


@deprecated
def sum_discrete_signals(ty1, ty2, max_dy=None):
    """
    ty1: Numpy array of size N1 x 2 where the first column is time and the
    second column is the first signal
    ty2: Numpy array of size N2 x 2 where the first column is time and the
    second column is the second signal

    ty1 and ty2 do not need to have the same dimensions (N1 does not have to
    equal N2). This function sums the linear interpolation of the two signals.

    tysum will have Nsum<=N1+N2 rows - signals will be summed at all time points
    from both ty1 and ty2 but not if there"""
    assert type(ty1) == np.ndarray and type(ty2) == np.ndarray
    assert ty1.ndim == 2 and ty2.ndim == 2  # confirm there is a t and y vector
    t1 = ty1[:, 0]
    y1 = ty1[:, 1]
    t2 = ty2[:, 0]
    y2 = ty2[:, 1]

    # get the sorted, unique values of t1 and t2
    tsum = np.sort(np.unique(np.append(t1, t2)))
    ysum = np.interp(tsum, t1, y1) + np.interp(tsum, t2, y2)

    if max_dy is not None:
        tsum, ysum = interp_limit_dy(tsum, ysum, max_dy)

    return np_smart_hstack(tsum, ysum)


@deprecated
def np_smart_hstack(x1, x2):
    """
    Quality of life function. Converts two (N,) numpy arrays or two lists into a
    (Nx2) array

    Example usage:
    a_list = [1,2,3,4]
    a_np_array = np.array([x**2 for x in a_list])
    np_smart_hstack(a_list, a_np_array)
    """
    onedim_types = [list, np.ndarray]
    assert type(x1) in onedim_types and type(x2) in onedim_types
    assert len(x1) == len(x2)  # confirm same size
    if type(x1) == list:
        x1 = np.array(x1)
    if type(x2) == list:
        x2 = np.array(x2)

    return np.hstack([x1.reshape(-1, 1), x2.reshape(-1, 1)])


@deprecated
def append_meshfunction_to_meshdomains(mesh, mesh_function):
    md = mesh.domains()
    mf_dim = mesh_function.dim()

    for idx, val in enumerate(mesh_function.array()):
        md.set_marker((idx, val), mf_dim)


# ====================================================
# I/O
# ====================================================

@deprecated
def write_sbmodel(filepath, pc, sc, cc, rc):
    """
    Takes a ParameterDF, SpeciesDF, CompartmentDF, and ReactionDF, and generates
    a .sbmodel file (a convenient concatenation of .json files with syntax
    similar to .xml)
    """

    # FIXME: This function cannot work due to undefined parameters
    f = open(filepath, "w")

    f.write("<sbmodel>\n")
    # parameters
    f.write("<parameters>\n")
    pdf.df.to_json(f)
    f.write("\n</parameters>\n")
    # species
    f.write("<species>\n")
    sdf.df.to_json(f)
    f.write("\n</species>\n")
    # compartments
    f.write("<compartments>\n")
    cdf.df.to_json(f)
    f.write("\n</compartments>\n")
    # reactions
    f.write("<reactions>\n")
    rdf.df.to_json(f)
    f.write("\n</reactions>\n")

    f.write("</sbmodel>\n")
    f.close()
    print(f"sbmodel file saved successfully as {filepath}!")


def read_sbmodel(filepath, output_type=dict):
    f = open(filepath, "r")
    lines = f.read().splitlines()
    if lines[0] != "<sbmodel>":
        raise Exception(f"Is {filepath} a valid .sbmodel file?")

    p_string = []
    c_string = []
    s_string = []
    r_string = []
    line_idx = 0

    while True:
        if line_idx >= len(lines):
            break
        line = lines[line_idx]
        if line == "</sbmodel>":
            print("Finished reading in sbmodel file")
            break

        if line == "<parameters>":
            print("Reading in parameters")
            while True:
                line_idx += 1
                if lines[line_idx] == "</parameters>":
                    break
                p_string.append(lines[line_idx])

        if line == "<species>":
            print("Reading in species")
            while True:
                line_idx += 1
                if lines[line_idx] == "</species>":
                    break
                s_string.append(lines[line_idx])

        if line == "<compartments>":
            print("Reading in compartments")
            while True:
                line_idx += 1
                if lines[line_idx] == "</compartments>":
                    break
                c_string.append(lines[line_idx])

        if line == "<reactions>":
            print("Reading in reactions")
            while True:
                line_idx += 1
                if lines[line_idx] == "</reactions>":
                    break
                r_string.append(lines[line_idx])

        line_idx += 1

    pdf = pandas.read_json("".join(p_string)).sort_index()
    sdf = pandas.read_json("".join(s_string)).sort_index()
    cdf = pandas.read_json("".join(c_string)).sort_index()
    rdf = pandas.read_json("".join(r_string)).sort_index()
    pc = stubs.model_assembly.ParameterContainer(nan_to_none(pdf))
    sc = stubs.model_assembly.SpeciesContainer(nan_to_none(sdf))
    cc = stubs.model_assembly.CompartmentContainer(nan_to_none(cdf))
    rc = stubs.model_assembly.ReactionContainer(nan_to_none(rdf))

    if output_type == dict:
        return {
            "parameter_container": pc,
            "species_container": sc,
            "compartment_container": cc,
            "reaction_container": rc,
        }
    elif output_type == tuple:
        return (pc, sc, cc, rc)


def pint_unit_to_quantity(pint_unit):
    if not isinstance(pint_unit, pint.Unit):
        raise TypeError("Input must be a pint unit")
    # returning pint.Quantity(1, pint_unit) changes the unit registry which we do NOT want
    return 1.0 * pint_unit


def pint_quantity_to_unit(pint_quantity):
    if not isinstance(pint_quantity, pint.Quantity):
        raise TypeError("Input must be a pint quantity")
    if pint_quantity.magnitude != 1.0:
        raise ValueError(
            "Trying to convert a pint quantity into a unit with magnitude != 1"
        )
    return pint_quantity.units


@deprecated
def convert_xml_to_hdf5(xml_filename, hdf5_filename, metadata_dims=None):
    if metadata_dims is None:
        metadata_dims = []
    else:
        assert all([dim in [0, 1, 2, 3] for dim in metadata_dims])

    # write
    mesh = d.Mesh(xml_filename)
    hdf5 = d.HDF5File(mesh.mpi_comm(), hdf5_filename, "w")
    hdf5.write(mesh, "/mesh")
    # write mesh functions
    for dim in metadata_dims:
        mf = d.MeshFunction("size_t", mesh, dim, value=mesh.domains())
        hdf5.write(mf, f"/mf{dim}")
    hdf5.close()


@deprecated
def read_hdf5(hdf5_filename, metadata_dims=None):
    if metadata_dims is None:
        metadata_dims = []
    else:
        assert all([dim in [0, 1, 2, 3] for dim in metadata_dims])

    # read
    mesh = d.Mesh()
    hdf5 = d.HDF5File(mesh.mpi_comm(), hdf5_filename, "r")
    hdf5.read(mesh, "/mesh", False)

    mfs = dict()
    for dim in metadata_dims:
        mfs[dim] = d.MeshFunction("size_t", mesh, dim)
        hdf5.read(mfs[dim], f"/mf{dim}")

    return mesh, mfs


def data_path():
    "data path for stubs directory"
    path = Path(".").resolve()
    subdir = "data"
    while True:
        if path.parts[-1] == "stubs" and path.joinpath(subdir).is_dir():
            path = path.joinpath(subdir)
            break
        path = path.parent
    return path


# Write a stopwatch class to measure time elapsed with a start, stop, and pause methods
# Keep track of timings in a list of lists called self.timings, each time the timer is paused,
# the time elapsed since the last pause is added to the sublist. Using stop resets the timer to zero
# and beings a new list of timings.

class Stopwatch:
    "Basic stopwatch class with inner/outer timings (pause and stop)"

    def __init__(
        self, name=None, time_unit="s", print_buffer=0, filename=None, start=False
    ):
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
            _fancy_print(
                f"{self.name} (iter {len(self._pause_timings)}) finished in {self.time_str(self._pause_timings[-1])} {self.time_unit}",
                format_type="logred",
                filename=self.filename,
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
            _fancy_print(
                f"{self._print_name} finished in {self.time_str(total_time)} {self.time_unit}",
                format_type="logred",
                filename=self.filename,
            )

        # for idx, t in enumerate(self._pause_timings):
        #     _fancy_print(f"{self.name} pause timings:", format_type='logred')
        #     _fancy_print(f"{self.name} {self.time_str(t)} {self.time_unit}", format_type='logred')

        # reset
        self.pause_timings.append(self._pause_timings)
        self._pause_timings = []
        self._times = []

    def set_timing(self, timing):
        self.stop_timings.append(timing)
        _fancy_print(
            f"{self._print_name} finished in {self.time_str(timing)} {self.time_unit}",
            format_type="logred",
            filename=self.filename,
        )

    def print_last_stop(self):
        _fancy_print(
            f"{self._print_name} finished in {self.time_str(self.stop_timings[-1])} {self.time_unit}",
            format_type="logred",
            filename=self.filename,
        )

    def time_str(self, t):
        return str({"us": 1e6, "ms": 1e3, "s": 1, "min": 1 / 60}[self.time_unit] * t)[
            0:8
        ]


@deprecated
def find_steady_state(
    reaction_list, constraints=None, return_equations=False, filename=None
):
    """
    Find the steady state of a list of reactions + constraints.
    """
    all_equations = list()
    all_species = set()
    for r in reaction_list:
        eqn = r.get_steady_state_equation()
        all_equations.append(eqn)
        all_species = all_species.union(eqn.free_symbols)

    num_eqns = len(all_equations)
    num_unknowns = len(all_species)
    num_constraints_nom = num_unknowns - num_eqns
    if constraints is None:
        constraints = list()
    if not isinstance(constraints, list):
        constraints = [constraints]
    _fancy_print(
        f"System has {num_eqns} equations and {num_unknowns} unknowns.",
        filename=filename,
    )
    _fancy_print(
        f"{len(constraints)} constraints provided. Requires {num_constraints_nom} constraints to be determined",
        format_type="log",
        filename=filename,
    )
    if num_constraints_nom != len(constraints):
        _fancy_print(
            f"Warning: system may be under or overdetermined.",
            format_type="log",
            filename=filename,
        )

    all_equations.extend(constraints)

    if return_equations:
        return all_equations, all_species
    else:
        return sympy.solve(all_equations, all_species)


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
        timestamp = datetime.now(timezone("US/Pacific")).strftime(
            "[%Y-%m-%d time=%H:%M:%S]"
        )
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
        title_str = (
            f"{colored(title_text, text_color)} {buffer(buffer_size*2+1+parity)}"
        )
    else:
        title_str = f"{buffer(buffer_size)} {colored(title_text, text_color)} {buffer(buffer_size+parity)}"
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
    """Given a facet and cell mesh function, return the topology of the face"""
    localCells = [mf3.array()[c.index()] for c in d.cells(f)]  # cells adjacent face
    if len(localCells) == 1:
        topology = "boundary"  # boundary facet
    elif len(localCells) == 2 and localCells[0] == localCells[1]:
        topology = "internal"  # internal facet
    elif len(localCells) == 2:
        topology = "interface"  # interface facet
    else:
        raise Exception("Face has more than two cells")
    return (topology, localCells)


@deprecated
def zplane_condition(cell, z=0.5):
    return cell.midpoint().z() > z + d.DOLFIN_EPS


def cube_condition(cell, xmin=0.3, xmax=0.7):
    return (
        (xmin - d.DOLFIN_EPS < cell.midpoint().x() < xmax + d.DOLFIN_EPS)
        and (xmin - d.DOLFIN_EPS < cell.midpoint().y() < xmax + d.DOLFIN_EPS)
        and (xmin - d.DOLFIN_EPS < cell.midpoint().z() < xmax + d.DOLFIN_EPS)
    )


def DemoCuboidsMesh(N=16, condition=cube_condition):
    """
    Creates a mesh for use in examples that contains two distinct cuboid subvolumes with a shared interface surface.
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


def write_mesh(mesh, mf2, mf3, filename="DemoCuboidMesh"):
    # Write mesh and meshfunctions to file
    hdf5 = d.HDF5File(mesh.mpi_comm(), filename + ".h5", "w")
    hdf5.write(mesh, "/mesh")
    hdf5.write(mf3, f"/mf3")
    hdf5.write(mf2, f"/mf2")
    # For visualization of domains
    d.File(filename + "_mf3.pvd") << mf3
    d.File(filename + "_mf2.pvd") << mf2


def nan_to_none(df):
    return df.replace({np.nan: None})
