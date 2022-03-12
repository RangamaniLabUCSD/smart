"""
General functions: array manipulation, data i/o, etc
"""
import pandas
import dolfin as d
#import trimesh
import numpy as np
import sympy
import scipy.interpolate as interp
import sys
import os
import re
import pint
from termcolor import colored
import stubs
gset = stubs.config.global_settings
from pandas import read_json
from contextlib import contextmanager as _contextmanager
from pathlib import Path
import time

comm = d.MPI.comm_world
rank = comm.rank
size = comm.size
root = 0



def sub(func, idx, collapse_function_space=True):
    "A collection of the proper ways to refer to a (sub) function, functionspace, etc. in dolfin"
    if isinstance(func, (list, tuple)):
        return func[idx]
    
    if isinstance(func, d.Function):
        # MixedFunctionSpace
        if func._functions:
            assert func.sub(idx) is func._functions[idx]
            return func.sub(idx)
            # if func.num_sub_spaces() <= 1 and idx == 0:
            #     # instead of just returning func, testing func.sub(0). u from W=MixedFunctionSpace is missing
            #     # a lot of information, even if W is defined from a single subspace, W = d.MixedFunctionSpace(V)
            #     return func.sub(0)
            # else:
            #     # should be equivalent to func._functions[idx]
            #     assert func.sub(idx) is func._functions[idx]
            #     return func.sub(idx)
        else:
            # scalar FunctionSpace
            if func.num_sub_spaces() <= 1 and idx == 0:
                return func
            else: 
                # Use d.split() to get subfunctions from a VectorFunctionSpace that can be used in forms
                return func.sub(idx)

    if isinstance(func, d.MixedFunctionSpace):
        return func.sub(idx)
        
    if isinstance(func, d.FunctionSpace):
        if func.num_sub_spaces() <= 1 and idx == 0:
            return func
        else:
            if collapse_function_space:
                return func.sub(idx).collapse()
            else:
                return func.sub(idx)
    
    if isinstance(func, d.function.argument.Argument):
        if func.function_space().num_sub_spaces() <= 1 and idx == 0:
            return func
        else:
            return func[idx]
    
    raise AssertionError()


# pandas
class ref:
    """
    Pandas dataframe doesn't play nice with dolfin indexed functions since it will try
    to take the length but dolfin will not return anything. We use ref() to trick pandas
    into treating the dolfin indexed function as a normal object rather than something
    with an associated length
    """
    def __init__(self, obj): self.obj = obj
    def get(self):    return self.obj
    def set(self, obj):      self.obj = obj

def insert_dataframe_col(df, columnName, columnNumber=None, items=None):
    """
    pandas requires some weird manipulation to insert some things, e.g. dictionaries, as dataframe entries
    items is a list of things
    by default, will insert a column of empty dicts as the last column
    """
    if items == None:
        items = [{} for x in range(df.shape[0])]
    if columnNumber == None:
        columnNumber = df.columns.size
    df.insert(columnNumber, columnName, items)

def nan_to_none(df):
    return df.replace({np.nan: None})

def submesh_dof_to_mesh_dof(Vsubmesh, submesh, bmesh_emap_0, V, submesh_species_index=0, mesh_species_index=0, index=None):
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


# dolfin DOF indices are not guaranteed to be equivalent to the vertex indices of the corresponding mesh
def submesh_dof_to_vertex(Vsubmesh, species_index, index=None):
    num_species = Vsubmesh.num_sub_spaces()
    if num_species == 0: num_species = 1
    num_dofs = int(len(Vsubmesh.dofmap().dofs())/num_species)
    if index == None:
        index = range(num_dofs)

    mapping = d.dof_to_vertex_map(Vsubmesh)
    mapping = mapping[range(species_index,len(mapping),num_species)] / num_species
    mapping = [int(x) for x in mapping]

    return [mapping[x] for x in index]

def submesh_to_bmesh(submesh, index):
    submesh_to_bmesh_vertex = submesh.data().array("parent_vertex_indices", 0)
    return submesh_to_bmesh_vertex[index]

def bmesh_to_parent(bmesh_emap_0, index):
    if bmesh_emap_0.max() > 1e9: # unless the mesh has 1e9 vertices this is a sign of an error
        raise Exception("Error in bmesh_emap.")
    return bmesh_emap_0[index]

def mesh_vertex_to_dof(V, species_index, index):
    num_species = V.num_sub_spaces()
    if num_species == 0: num_species = 1

    mapping = d.vertex_to_dof_map(V)

    mapping = mapping[range(species_index, len(mapping), num_species)]

    return [mapping[x] for x in index]

def round_to_n(x,n):
    """
    Rounds to n sig figs
    """
    if x == 0:
        return 0
    else:
        sign = np.sign(x)
        x = np.abs(x)
        return sign*round(x, -int(np.floor(np.log10(x))) + (n - 1))


def interp_limit_dy(t,y,max_dy,interp_type='linear'):
    """
    Interpolates t and y such that dy between time points is never greater than max_dy
    Maintains all original t and y
    """
    interp_t = t.reshape(-1,1)
    dy_vec = y[1:] - y[:-1]

    for idx, dy in enumerate(dy_vec):
        npoints = np.int(np.ceil(np.abs(dy/max_dy))) - 1
        if npoints >= 1:
            new_t = np.linspace(t[idx], t[idx+1], npoints+2)[1:-1]
            interp_t = np.vstack([interp_t, new_t.reshape(-1,1)])

    interp_t = np.sort(interp_t.reshape(-1,))
    interp_y = interp.interp1d(t,y,kind=interp_type)(interp_t)

    return (interp_t, interp_y)

def sum_discrete_signals(ty1, ty2, max_dy=None):
    """
    ty1: Numpy array of size N1 x 2 where the first column is time and the
    second column is the first signal
    ty2: Numpy array of size N2 x 2 where the first column is time and the
    second column is the second signal

    ty1 and ty2 do not need to have the same dimensions (N1 does not have to
    equal N2). This function sums the linear interpolation of the two signals.

    tysum will have Nsum<=N1+N2 rows - signals will be summed at all time points
    from both ty1 and ty2 but not if there """
    assert type(ty1)==np.ndarray and type(ty2)==np.ndarray
    assert ty1.ndim==2 and ty2.ndim==2 # confirm there is a t and y vector
    t1 = ty1[:,0]; y1 = ty1[:,1]; t2 = ty2[:,0]; y2 = ty2[:,1]

    # get the sorted, unique values of t1 and t2
    tsum = np.sort(np.unique(np.append(t1,t2)))
    ysum = np.interp(tsum, t1, y1) + np.interp(tsum, t2, y2)

    if max_dy is not None:
        tsum, ysum = interp_limit_dy(tsum, ysum, max_dy)

    return np_smart_hstack(tsum, ysum)


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
    assert len(x1) == len(x2) # confirm same size
    if type(x1) == list: x1 = np.array(x1)
    if type(x2) == list: x2 = np.array(x2)

    return np.hstack([x1.reshape(-1,1), x2.reshape(-1,1)])

def append_meshfunction_to_meshdomains(mesh, mesh_function):
    md = mesh.domains()
    mf_dim = mesh_function.dim()

    for idx, val in enumerate(mesh_function.array()):
        md.set_marker((idx,val), mf_dim)

# def color_print(full_text, color):
#     if rank==root:
#         split_text = [s for s in re.split('(\n)', full_text) if s] # colored doesn't like newline characters
#         for text in split_text:
#             if text == '\n':
#                 print()
#             else:
#                 print(colored(text, color=color))

# ====================================================
# fancy printing
# ====================================================
def _fancy_print(title_text, buffer_color=None, text_color=None, filler_char=None,
                             num_banners=None, new_lines=None, left_justify=None,
                             format_type='default', filename=None):
    "Formatted text to stand out."

    # Initialize with the default options
    buffer_color_ = 'cyan'; text_color_ = 'green'; filler_char_ = '='
    num_banners_ = 0; new_lines_ = [0,0]; left_justify_ = False
    # Override with format_type options
    if format_type == 'default':
        pass
    elif format_type == 'title':
        text_color_ = 'magenta'; num_banners_ = 1; new_lines_ = [1,0]
    elif format_type == 'subtitle':
        text_color_ = 'green'; filler_char_ = '.'; left_justify_=True
    elif format_type == 'data':
        buffer_color_='white'; text_color_ = 'white'; filler_char_ = ''; left_justify_=True
    elif format_type == 'data_important':
        buffer_color_='white'; text_color_ = 'red'; filler_char_ = ''; left_justify_=True
    elif format_type == 'log':
        buffer_color_='white'; text_color_ = 'green'; filler_char_ = ''; left_justify_=True
    elif format_type == 'logred':
        buffer_color_='white'; text_color_ = 'red'; filler_char_ = ''; left_justify_=True
    elif format_type == 'log_important':
        buffer_color_='white'; text_color_ = 'magenta'; filler_char_ = '.'
    elif format_type == 'log_urgent':
        buffer_color_='white'; text_color_ = 'red'; filler_char_ = '.'
    elif format_type == 'warning':
        buffer_color_='magenta'; text_color_ = 'red'; filler_char_ = '!'; num_banners_=2; new_lines_=[1,1]
    elif format_type == 'timestep':
        text_color_ = 'red'; num_banners_ = 2; filler_char_ = '.'; new_lines_=[1,1]
    elif format_type == 'solverstep':
        text_color_ = 'red'; num_banners_ = 1; filler_char_ = '.'; new_lines_=[1,1]
    elif format_type == 'assembly':
        text_color_ = 'magenta'; num_banners_ = 0; filler_char_ = '.'; new_lines_=[1,0]
    elif format_type == 'assembly_sub':
        text_color_ = 'magenta'; num_banners_ = 0; filler_char_ = ''; new_lines_=[0,0]; left_justify_=True
    elif format_type is not None:
        raise ValueError("Unknown formatting_type.")

    # Override again with user options
    if buffer_color is None: buffer_color = buffer_color_
    if text_color is None: text_color = text_color_
    if filler_char is None: filler_char = filler_char_
    if num_banners is None: num_banners = num_banners_
    if new_lines is None: new_lines = new_lines_
    if left_justify is None: left_justify = left_justify_
    
    # include MPI rank in message
    if size > 1:
        title_text = f"CPU {rank}: {title_text}"

    # calculate optimal buffer size
    min_buffer_size = 5
    terminal_width = 120
    buffer_size = max([min_buffer_size, int((terminal_width-1 - len(title_text))/2 - 1)]) # terminal width == 80
    title_str_len = (buffer_size+1)*2 + len(title_text)
    parity=1 if title_str_len==78 else 0

    # color/stylize buffer, text, and banner
    buffer = lambda buffer_size: colored(filler_char*buffer_size, buffer_color)
    if left_justify:
        title_str=f"{colored(title_text, text_color)} {buffer(buffer_size*2+1+parity)}"
    else:
        title_str = f"{buffer(buffer_size)} {colored(title_text, text_color)} {buffer(buffer_size+parity)}"
    banner = colored(filler_char*(title_str_len+parity), buffer_color)

    def print_out(text, filename=None):
        "print to file and terminal"
        if filename is not None:
            with open(filename, 'a') as f:
                f.write(text+'\n')
        elif stubs.config.global_settings['log_filename'] is not None:
            with open(stubs.config.global_settings['log_filename'], 'a') as f:
                f.write(text+'\n')
        print(text)
        
    # initial spacing
    if new_lines[0] > 0: print_out('\n'*(new_lines[0]-1), filename)
    # print first banner
    for _ in range(num_banners):
        print_out(f"{banner}", filename)
    # print main text
    print_out(title_str, filename)
    # print second banner
    for _ in range(num_banners):
        print_out(f"{banner}", filename)
    # end spacing
    if new_lines[1] > 0: print_out('\n'*(new_lines[1]-1), filename)

# # demonstrate built in options
# def _fancy_print_options():
#     for format_type in ['title', 'subtitle', 'log', 'log_important', 'log_urgent', 'timestep', 'solverstep']:
#         _fancy_print(format_type, format_type=format_type)

# ====================================================
# I/O
# ====================================================

def json_to_ObjectContainer(json_str, data_type=None):
    """
    Converts a json_str (either a string of the json itself, or a filepath to
    the json)
    """
    if not data_type:
        raise Exception("Please include the type of data this is (parameters, species, compartments, reactions).")

    if json_str[-5:] == '.json':
        if not os.path.exists(json_str):
            raise Exception("Cannot find JSON file, %s"%json_str)   

    df = read_json(json_str).sort_index()
    df = nan_to_none(df)
    if data_type in ['parameters', 'parameter', 'param', 'p']:
        return stubs.model_assembly.ParameterContainer(df)
    elif data_type in ['species', 'sp', 'spec', 's']:
        return stubs.model_assembly.SpeciesContainer(df)
    elif data_type in ['compartments', 'compartment', 'comp', 'c']:
        return stubs.model_assembly.CompartmentContainer(df)
    elif data_type in ['reactions', 'reaction', 'r', 'rxn']:
        return stubs.model_assembly.ReactionContainer(df)
    else:
        raise Exception("I don't know what kind of ObjectContainer this .json file should be")



# def write_sbmodel(filepath, pdf, sdf, cdf, rdf):
#     """
#     Takes a ParameterDF, SpeciesDF, CompartmentDF, and ReactionDF, and generates
#     a .sbmodel file (a convenient concatenation of .json files with syntax
#     similar to .xml)
#     """
#     f = open(filepath, "w")

#     f.write("<sbmodel>\n")
#     # parameters
#     f.write("<parameters>\n")
#     pdf.df.to_json(f)
#     f.write("\n</parameters>\n")
#     # species
#     f.write("<species>\n")
#     sdf.df.to_json(f)
#     f.write("\n</species>\n")
#     # compartments
#     f.write("<compartments>\n")
#     cdf.df.to_json(f)
#     f.write("\n</compartments>\n")
#     # reactions
#     f.write("<reactions>\n")
#     rdf.df.to_json(f)
#     f.write("\n</reactions>\n")

#     f.write("</sbmodel>\n")
#     f.close()
#     print(f"sbmodel file saved successfully as {filepath}!")


def write_sbmodel(filepath, pc, sc, cc, rc):
    """
    Takes a ParameterDF, SpeciesDF, CompartmentDF, and ReactionDF, and generates
    a .sbmodel file (a convenient concatenation of .json files with syntax
    similar to .xml)
    """
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
        if line == '</sbmodel>':
            print("Finished reading in sbmodel file")
            break

        if line == '<parameters>':
            print("Reading in parameters")
            while True:
                line_idx += 1
                if lines[line_idx] == '</parameters>':
                    break
                p_string.append(lines[line_idx])

        if line == '<species>':
            print("Reading in species")
            while True:
                line_idx += 1
                if lines[line_idx] == '</species>':
                    break
                s_string.append(lines[line_idx])

        if line == '<compartments>':
            print("Reading in compartments")
            while True:
                line_idx += 1
                if lines[line_idx] == '</compartments>':
                    break
                c_string.append(lines[line_idx])

        if line == '<reactions>':
            print("Reading in reactions")
            while True:
                line_idx += 1
                if lines[line_idx] == '</reactions>':
                    break
                r_string.append(lines[line_idx])

        line_idx += 1

    pdf = pandas.read_json(''.join(p_string)).sort_index()
    sdf = pandas.read_json(''.join(s_string)).sort_index()
    cdf = pandas.read_json(''.join(c_string)).sort_index()
    rdf = pandas.read_json(''.join(r_string)).sort_index()
    pc = stubs.model_assembly.ParameterContainer(nan_to_none(pdf))
    sc = stubs.model_assembly.SpeciesContainer(nan_to_none(sdf))
    cc = stubs.model_assembly.CompartmentContainer(nan_to_none(cdf))
    rc = stubs.model_assembly.ReactionContainer(nan_to_none(rdf))

    if output_type==dict:
        return {'parameter_container': pc,   'species_container': sc, 
                'compartment_container': cc, 'reaction_container': rc}
    elif output_type==tuple:
        return (pc, sc, cc, rc)

# def create_sbmodel(p, s, c, r, output_type=dict):
#     pc = stubs.model_assembly.ParameterContainer(p)
#     sc = stubs.model_assembly.SpeciesContainer(s)
#     cc = stubs.model_assembly.CompartmentContainer(c)
#     rc = stubs.model_assembly.ReactionContainer(r)

#     if output_type==dict:
#         return {'parameter_container': pc,   'species_container': sc, 
#                 'compartment_container': cc, 'reaction_container': rc}
#     elif output_type==tuple:
#         return (pc, sc, cc, rc)

def empty_sbmodel():
    pc = stubs.model_assembly.ParameterContainer()
    sc = stubs.model_assembly.SpeciesContainer()
    cc = stubs.model_assembly.CompartmentContainer()
    rc = stubs.model_assembly.ReactionContainer()
    return pc, sc, cc, rc 

def pint_unit_to_quantity(pint_unit):
    if not isinstance(pint_unit, pint.Unit):
        raise TypeError("Input must be a pint unit")
    # returning pint.Quantity(1, pint_unit) changes the unit registry which we do NOT want
    return 1.0*pint_unit 

def pint_quantity_to_unit(pint_quantity):
    if not isinstance(pint_quantity, pint.Quantity):
        raise TypeError("Input must be a pint quantity")
    if pint_quantity.magnitude != 1.0:
        raise ValueError("Trying to convert a pint quantity into a unit with magnitude != 1")
    return pint_quantity.units


# # Some stack exchange code to redirect/suppress c++ stdout
# # https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262#22434262
# def _fileno(file_or_fd):
#     fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
#     if not isinstance(fd, int):
#         raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
#     return fd

# @_contextmanager
# def _stdout_redirected(to=os.devnull, stdout=None):
#     if stdout is None:
#        stdout = sys.stdout

#     stdout_fd = _fileno(stdout)
#     # copy stdout_fd before it is overwritten
#     #NOTE: `copied` is inheritable on Windows when duplicating a standard stream
#     with os.fdopen(os.dup(stdout_fd), 'wb') as copied: 
#         stdout.flush()  # flush library buffers that dup2 knows nothing about
#         try:
#             os.dup2(_fileno(to), stdout_fd)  # $ exec >&to
#         except ValueError:  # filename
#             with open(to, 'wb') as to_file:
#                 os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
#         try:
#             yield stdout # allow code to be run with the redirected stdout
#         finally:
#             # restore stdout to its previous value
#             #NOTE: dup2 makes stdout_fd inheritable unconditionally
#             stdout.flush()
#             os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied
    

def convert_xml_to_hdf5(xml_filename, hdf5_filename, metadata_dims=None):
    if metadata_dims is None:
        metadata_dims = []
    else:
        assert all([dim in [0,1,2,3] for dim in metadata_dims])

    # write
    mesh = d.Mesh(xml_filename)
    hdf5 = d.HDF5File(mesh.mpi_comm(), hdf5_filename, 'w')
    hdf5.write(mesh, '/mesh')
    # write mesh functions
    for dim in metadata_dims:
        mf = d.MeshFunction('size_t', mesh, dim, value=mesh.domains())
        hdf5.write(mf, f"/mf{dim}")
    hdf5.close()

def read_hdf5(hdf5_filename, metadata_dims=None):
    if metadata_dims is None:
        metadata_dims = []
    else:
        assert all([dim in [0,1,2,3] for dim in metadata_dims])
    
    # read
    mesh = d.Mesh()
    hdf5 = d.HDF5File(mesh.mpi_comm(), hdf5_filename, 'r')
    hdf5.read(mesh, '/mesh', False)

    mfs = dict()
    for dim in metadata_dims:
        mfs[dim] = d.MeshFunction('size_t', mesh, dim)
        hdf5.read(mfs[dim], f"/mf{dim}")

    return mesh, mfs
    
# def fix_mesh_normals(dolfin_mesh):
#     assert isinstance(dolfin_mesh, d.Mesh)
#     tdim = dolfin_mesh.topology().dim()
#     assert tdim in [2,3]
#     if tdim == 3:
#         print(f"Input mesh has topological dimension 3. Fixing normals of boundary mesh instead.")
#         mesh = d.BoundaryMesh(dolfin_mesh, 'exterior')
#     else:
#         mesh = dolfin_mesh
    
#     triangles = mesh.cells()
#     vertices  = mesh.vertices()
#     tmesh = trimesh.Trimesh(vertices, triangles, process=False)
#     tmesh.fix_normals()
#     tmesh.export()


def data_path():
    "data path for stubs directory"
    path    = Path('.').resolve()
    subdir  = 'data'
    while True:
        if path.parts[-1]=='stubs' and path.joinpath(subdir).is_dir():
            path = path.joinpath(subdir)
            break
        path = path.parent
    return path



# Write a stopwatch class to measure time elapsed with a start, stop, and pause methods
# Keep track of timings in a list of lists called self.timings, each time the timer is paused,
# the time elapsed since the last pause is added to the sublist. Using stop resets the timer to zero
# and beings a new list of timings.

class Stopwatch():
    "Basic stopwatch class with inner/outer timings (pause and stop)"
    def __init__(self, name=None, time_unit='s', print_buffer=0, filename=None):
        self.name = name
        self.time_unit = time_unit
        self.stop_timings = []  # length = number of stops
        self.pause_timings = [] # length = number of stops (list of lists)
        self._pause_timings = [] # length = number of pauses (reset on stop)
        self._times = []
        self.is_paused = True
        self.print_buffer=print_buffer
        self._print_name = f"{str(self.name): <{self.print_buffer}}"
        #self.start()
        self.filename=filename
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
            _fancy_print(f"{self.name} (iter {len(self._pause_timings)}) finished in {self.time_str(self._pause_timings[-1])} {self.time_unit}", format_type='logred', filename=self.filename)
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
            _fancy_print(f"{self._print_name} finished in {self.time_str(total_time)} {self.time_unit}", format_type='logred', filename=self.filename)

        # for idx, t in enumerate(self._pause_timings):
        #     _fancy_print(f"{self.name} pause timings:", format_type='logred')
        #     _fancy_print(f"{self.name} {self.time_str(t)} {self.time_unit}", format_type='logred')

        # reset
        self.pause_timings.append(self._pause_timings)
        self._pause_timings = []
        self._times = []
    def set_timing(self, timing):
        self.stop_timings.append(timing)
        _fancy_print(f"{self._print_name} finished in {self.time_str(timing)} {self.time_unit}", format_type='logred', filename=self.filename)

    def print_last_stop(self):
        _fancy_print(f"{self._print_name} finished in {self.time_str(self.stop_timings[-1])} {self.time_unit}", format_type='logred', filename=self.filename)
        
    
    def time_str(self, t):
        return str({'us': 1e6, 'ms': 1e3, 's': 1, 'min': 1/60}[self.time_unit]*t)[0:8]
        
    
def find_steady_state(reaction_list, constraints=None, return_equations=False, filename=None):
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
    _fancy_print(f"System has {num_eqns} equations and {num_unknowns} unknowns.", filename=filename)
    _fancy_print(f"{len(constraints)} constraints provided. Requires {num_constraints_nom} constraints to be determined", format_type='log', filename=filename)
    if num_constraints_nom != len(constraints):
        _fancy_print(f"Warning: system may be under or overdetermined.", format_type='log', filename=filename)
    
    all_equations.extend(constraints)
    
    if return_equations:
        return all_equations, all_species
    else:
        return sympy.solve(all_equations, all_species)