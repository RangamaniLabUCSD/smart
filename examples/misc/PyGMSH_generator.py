"""
Idealized mesh for example codes generated using pygmsh
"""
import pygmsh
import meshio
import stubs
import dolfin as d


# construct a unit cube
g = pygmsh.built_in.Geometry()

corner = [0,0,0]
dim = [1,1,1]
num_vert = 2**len(dim)
g.add_box(corner[0], corner[0]+dim[0], corner[1], corner[1]+dim[1], corner[2], corner[2]+dim[2], lcar=0.05)


m = pygmsh.generate_mesh(g, dim=3, prune_vertices=False) # setting prune_vertices=False; pygmsh fails otherwise
#m = pygmsh.generate_mesh(g, dim=2, prune_vertices=False,prune_z_0=True) # setting prune_vertices=False; pygmsh fails otherwise
meshio.write("temp_mesh.xml", m)

# mark volume/boundaries
mesh = d.Mesh("temp_mesh.xml")
mf = d.MeshFunction("size_t", mesh, 3, value=1)
stubs.common.append_meshfunction_to_meshdomains(mesh, mf)

# write out mesh with appended mesh function information
d.File("unit_cube.xml") << mesh
d.File("unit_cube.pvd") << mf
