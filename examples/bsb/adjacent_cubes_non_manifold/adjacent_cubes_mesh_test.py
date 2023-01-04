#adjacent_cubes_mesh_test.py
from copy import deepcopy

import dolfin as d
import numpy as np


def get_bmc_to_pmf_map(bm, sm, pm, bmc_to_smf, smv_to_pmv):
    """
    For a BoundaryMesh created from a SubMesh of a Parent Mesh, get map of
    BoundaryMesh cells to ParentMesh facets.
    from BoundaryMesh to ParentMesh
    c=cell (dim=n), f=facet (dim=n-1), v=vertex (dim=0)

    all indices are assumed to be local unless stated otherwise
     * "sm_farray_up" implies indices are from the mesh one level above it
    """
    # Useful lambdas
    facet_array             = lambda mesh: np.array([x.entities(0) for x in d.facets(mesh)])
    facet_midpoint          = lambda mesh, fidx: list(d.facets(mesh))[fidx].midpoint().array()
    cell_midpoint           = lambda mesh, cidx: list(d.cells(mesh))[cidx].midpoint().array()
    vidx_to_coord           = lambda mesh, vidx: list(d.vertices(mesh))[vidx].point().array()
    cellidx_to_barycenter   = lambda mesh, cidx: mesh.coordinates()[mesh.cells()[cidx,:],:].mean(axis=0)

    # initialize mapper
    mapper = list()

    # facet arrays for SubMesh and ParentMesh
    sm_farray           = facet_array(sm) # array of vertex indices for each facet
    temp_sm_farray_up   = np.array([smv_to_pmv[i] for i in sm_farray]) # array of parent mesh vertex indices for each facet
    temp_sort_cols      = np.argsort(temp_sm_farray_up, axis=1)
    sm_farray_up        = np.take_along_axis(temp_sm_farray_up, temp_sort_cols, axis=1) # sort so each row is in ascending order
    pm_farray           = facet_array(pm)

    # loop over BoundaryMesh cells (facets of SubMesh and ParentMesh)
    for bm_cidx in range(bm.num_cells()):
        sm_fidx = bmc_to_smf[bm_cidx] # submesh facet index (cell of bmesh is facet of submesh)
        # find which parent mesh facet has same vertex indices as submesh facet
        pm_fidx = np.where((pm_farray==sm_farray_up[sm_fidx,:]).all(axis=1))[0][0]
        # error check
        if pm_fidx.size != 1:
            raise ValueError(f"Could not find matching facet from Parent Mesh for BoundaryMesh facet {bm_cidx}")

        mapper.append(pm_fidx)

    return mapper


#def get_full_boundary_mesh(self):
#m       = self.dolfin_mesh
#dim     = self.dimensionality
pm  = d.Mesh('adjacent_cubes.xml')
dim = pm.geometric_dimension()

# Mesh function for sub-volumes
mf_vv = d.MeshFunction('size_t', pm, dim, pm.domains())
# Get unique sub-volume markers
unique_svmarkers = list(set(mf_vv.array()))
# Get BoundaryMesh for each sub-volume
sm = dict() # submeshes for each sub-volume
bm = dict() # boundary mesh for each submesh
# Entity maps
smv_to_pmv  = dict()
bmv_to_smv  = dict()
bmc_to_smf  = dict()
bmc_to_pmf  = dict()


# Get SubMeshes, BoundaryMeshes of SubMeshes, and entity maps between the different levels of meshes
for svmarker in unique_svmarkers:
    # Get SubMesh (same dimensionality as ParentMesh)
    sm[svmarker] = d.SubMesh(pm, mf_vv, svmarker)
    # Get entity maps for SubmMesh -> ParentMesh
    smv_to_pmv[svmarker] = sm[svmarker].data().array('parent_vertex_indices', 0)
    # Get BoundaryMesh of SubMesh
    bm[svmarker] = d.BoundaryMesh(sm[svmarker], 'exterior')
#    print(f"BoundaryMesh of SubMesh (marker {svmarker}) has {bm[svmarker].num_vertices()} vertices and {bm[svmarker].num_cells()} cells.")
#    print(f"BoundaryMesh of SubMesh (marker {svmarker}) has min,max z-value "
#          + f"({bm[svmarker].coordinates()[:,2].min()}, {bm[svmarker].coordinates()[:,2].max()})\n")
    # Get entity maps for BoundaryMesh -> SubMesh
    temp_emap_0 = bm[svmarker].entity_map(0)
    temp_emap_n = bm[svmarker].entity_map(dim-1)
    bmv_to_smv[svmarker] = deepcopy(temp_emap_0.array()) # maps from BoundaryMesh vertices to SubMesh vertices
    bmc_to_smf[svmarker] = deepcopy(temp_emap_n.array()) # maps from BoundaryMesh cells to SubMesh facets
#   print(f"BoundaryMesh of SubMesh (marker {svmarker}) has cell {cell_idx} with barycenter {cellidx_to_barycenter(bm[svmarker], cell_idx)}")
    # Combine entity maps to get a mapping from BoundaryMesh cells to SubMesh facets
    bmc_to_pmf[svmarker] = get_bmc_to_pmf_map(bm[svmarker], sm[svmarker], pm, bmc_to_smf[svmarker], smv_to_pmv[svmarker])



# Union of boundary meshes
initial_svmarker    = unique_svmarkers[0] # arbitrary, doesn't really matter
remaining_svmarkers = [svm for svm in unique_svmarkers if svm!=initial_svmarker]
# initialize BoundaryMesh union
bm_union = d.BoundaryMesh(d.SubMesh(pm, mf_vv, initial_svmarker), 'exterior')
# initialize unique facet markers in the ParentMesh frame
unique_pmf = deepcopy(bmc_to_pmf[initial_svmarker])

# Loop over other BoundaryMeshes and find/remove overlapping cells
for svmarker in remaining_svmarkers:
    for pmf in bmc_to_pmf[svmarker]:
        if pmf not in unique_pmf:
            print(f"Adding pmf {pmf} from BoundaryMesh with sub-volume marker {svmarker}.")
            unique_pmf.append(pmf)
            continue
        else:
            print(f"The pmf {pmf} from BoundaryMesh with sub-volume marker {svmarker} is not unique... skipping.")



# Use MeshEditor to combine BoundaryMeshes


# numpy array. row=facet_idx, columns=vertex_idx



## sanity check
#def quiver_plot_normal_vectors(mesh):
#    """
#    A sanity check to understand how FEniCS computes normal vectors for SubMeshes
#    """
#
#

#np_unique_array = lambda
=>>>>>>>>>>>>>>>>>>>>>>>>do this next. merge all vertices
#https://stackoverflow.com/questions/49950412/merge-two-numpy-arrays-and-delete-duplicates

# https://fenicsproject.org/qa/185/entity-mapping-between-a-submesh-and-the-parent-mesh/

# Get entity maps
temp_emap_0 =

# Find overlapping vertices and cells


# Combine meshes
me = d.MeshEditor()
b = bm[11]
btype = d.CellType.type2string(b.type().cell_type())
me.open(b, btype, b.topology().dim(), b.geometric_dimension())
