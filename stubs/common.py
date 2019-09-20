# ====================================================
# General python
import pandas as pd
import dolfin as d
import numpy as np

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
    return df.replace({pd.np.nan: None})




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


