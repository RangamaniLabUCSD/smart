import dolfin as d

class Mesh(object):
    def __init__(self, mesh_filename=None, name='parent_mesh', is_parent_mesh=False, dolfin_mesh=None):
        self.mesh_filename  = mesh_filename
        self.name           = name
        self.backend        = backend
        if mesh_filename is not None:
            self.load_mesh()
    def load_mesh(self):
        self.dolfin_mesh = d.Mesh(self.mesh_filename)
        print(f"Mesh, {self.name}, successfully loaded from file: {self.mesh_filename}!")

    #def extract_submeshes(self):
    #    """
    #    The parent mesh is the full mesh being simulated on and may include multiple volume/surface meshes delineated by
    #    marker values.
    #    """
    #    if self.dolfin_mesh is None:
    #        raise ValueError("Load in mesh with load_mesh() first.")
    #    elif self.is_parent_mesh is not True:
    #        raise ValueError(f"Mesh {name} must be a parent_mesh in order to extract submeshes.")
    #    else:
    #        # Get minimum and maximum dimensions of meshes being computed on.
    #        volumeDim  = self.dolfin_mesh.geometric_dimension()
    #        volumeDim  = self.max_dim
    #        surfaceDim = self.min_dim
    #        if (volumeDim - surfaceDim) not in [0,1]:
    #            raise ValueError("(Highest mesh dimension - smallest mesh dimension) must be either 0 or 1.")

    #        # Get volume and boundary mesh
    #        vmesh                           = self.meshes[main_mesh_str]
    #        self.Dict[main_mesh_str].mesh   = self.meshes[main_mesh_str]
    #        smesh                           = d.BoundaryMesh(vmesh, "exterior")

    #        # When smesh.entity_map() is called together with .array() it will return garbage values. We should only call 
    #        # entity_map once to avoid this
    #        temp_emap_0  = smesh.entity_map(0)
    #        smesh_emap_0 = deepcopy(temp_emap_0.array()) # entity map to vertices
    #        temp_emap_n  = smesh.entity_map(surfaceDim)
    #        smesh_emap_n = deepcopy(temp_emap_n.array()) # entity map to facets

    #        # Mesh functions
    #        # 
    #        # vvmf: cell markers for volume mesh. Used to distinguish sub-volumes
    #        #
    #        #
    #        #
    #        #vvmf         = d.MeshFunction("size_t", vmesh, volumeDim, vmesh.domains())  # cell markers for volume mesh
    #        #vsmf         = d.MeshFunction("size_t", vmesh, surfaceDim, vmesh.domains()) # facet markers for volume mesh
    #        #smf          = d.MeshFunction("size_t", bmesh, surfaceDim)                  # cell markers for surface mesh
    #        #vmf_combined = d.MeshFunction("size_t", vmesh, surfaceDim, vmesh.domains()) # 
    #        #bmf_combined = d.MeshFunction("size_t", bmesh, surfaceDim)

    #        vmf          = d.MeshFunction("size_t", vmesh, surfaceDim, vmesh.domains())
    #        bmf          = d.MeshFunction("size_t", bmesh, surfaceDim)
    #        vmf_combined = d.MeshFunction("size_t", vmesh, surfaceDim, vmesh.domains())
    #        bmf_combined = d.MeshFunction("size_t", bmesh, surfaceDim)

    #        # iterate through facets of bmesh (transfer markers from volume mesh function to boundary mesh function)
    #        for idx, facet in enumerate(d.entities(bmesh,surfaceDim)): 
    #            vmesh_idx = bmesh_emap_n[idx] # get the index of the face on vmesh corresponding to this face on bmesh
    #            vmesh_boundarynumber = vmf.array()[vmesh_idx] # get the value of the mesh function at this face
    #            bmf.array()[idx] = vmesh_boundarynumber # set the value of the boundary mesh function to be the same value

    #        # combine markers for subdomains specified as a list of markers
    #        for comp_name, comp in self.Dict.items():
    #            if type(comp.cell_marker) == list:
    #                if not all([type(x)==int for x in comp.cell_marker]):
    #                    raise ValueError("Cell markers were given as a list but not all elements were ints.")

    #                first_index_marker = comp.cell_marker[0] # combine into the first marker of the list 
    #                comp.first_index_marker = first_index_marker
    #                Print(f"Combining markers {comp.cell_marker} (for component {comp_name}) into single marker {first_index_marker}.")
    #                for marker_value in comp.cell_marker:
    #                    vmf_combined.array()[vmf.array() == marker_value] = first_index_marker
    #                    bmf_combined.array()[bmf.array() == marker_value] = first_index_marker
    #            elif type(comp.cell_marker) == int:
    #                comp.first_index_marker = comp.cell_marker
    #                vmf_combined.array()[vmf.array() == comp.cell_marker] = comp.cell_marker
    #                bmf_combined.array()[bmf.array() == comp.cell_marker] = comp.cell_marker
    #            else:
    #                raise ValueError("Cell markers must either be provided as an int or list of ints")


    #        # Loop through compartments: extract submeshes and integration measures
    #        for comp_name, comp in self.Dict.items():
    #            # FEniCS doesn't allow parallelization of SubMeshes. We need
    #            # SubMeshes because one boundary will often have multiple domains of
    #            # interest with different species (e.g., PM, ER). By exporting the
    #            # submeshes in serial we can reload them back in in parallel.

    #            if comp_name!=main_mesh_str and comp.dimensionality==surfaceDim:
    #                # # TODO: fix this (parallel submesh)
    #                # if size > 1: # if we are running in parallel
    #                #     Print("CPU %d: Loading submesh for %s from file" % (rank, comp_name))
    #                #     submesh = d.Mesh(d.MPI.comm_self, 'submeshes/submesh_' + comp.name + '_' + str(comp.cell_marker) + '.xml')
    #                #     self.meshes[comp_name] = submesh
    #                #     comp.mesh = submesh
    #                # else:
    #                submesh = d.SubMesh(bmesh, bmf_combined, comp.first_index_marker)
    #                self.vertex_mappings[comp_name] = submesh.data().array("parent_vertex_indices", 0)
    #                self.meshes[comp_name] = submesh
    #                comp.mesh = submesh

    #                # # TODO: fix this (parallel submesh)
    #                # if save_to_file:
    #                #     Print("Saving submeshes %s for use in parallel" % comp_name)
    #                #     save_str = 'submeshes/submesh_' + comp.name + '_' + str(comp.cell_marker) + '.xml'
    #                #     d.File(save_str) << submesh

    #            # integration measures
    #            if comp.dimensionality==volumeDim:
    #                comp.ds = d.Measure('ds', domain=comp.mesh, subdomain_data=vmf_combined, metadata={'quadrature_degree': 3})
    #                comp.ds_uncombined = d.Measure('ds', domain=comp.mesh, subdomain_data=vmf, metadata={'quadrature_degree': 3})
    #                comp.dP = None
    #            elif comp.dimensionality<volumeDim:
    #                comp.dP = d.Measure('dP', domain=comp.mesh)
    #                comp.ds = None
    #            else:
    #                raise Exception(f"Internal error: {comp_name} has a dimension larger than then the volume dimension.")
    #            comp.dx = d.Measure('dx', domain=comp.mesh, metadata={'quadrature_degree': 3}) 
