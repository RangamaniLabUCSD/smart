# mixed assembly mwe
import dolfin as d
# Create a mesh
mesh = d.UnitCubeMesh(40,40,40)
z_slice = 0.5 # value of z we use to demaracate the different subvolumes
mesh.coordinates()[:,2] *= 4 # something about partitioning leads to hanging
mesh.init()

# define mesh functions
mf3 = d.MeshFunction("size_t", mesh, 3, 0)
mf2 = d.MeshFunction("size_t", mesh, 2, 0)
for c in d.cells(mesh):
    mf3[c] = ( c.midpoint().z() < z_slice+d.DOLFIN_EPS )*1

for f in d.facets(mesh):
    mf2[f] = (z_slice-d.DOLFIN_EPS <= f.midpoint().z() <= z_slice+d.DOLFIN_EPS)*1

# Create submeshes using MeshView
mesh0 = d.MeshView.create(mf3, 1); mesh_ = d.MeshView.create(mf2, 1)
# build mesh mapping
mesh_.build_mapping(mesh0)
# function space and function
V0 = d.FunctionSpace(mesh0, "CG", 1)
u0 = d.Function(V0)
# measures
dx = d.Measure('dx', domain=mesh)
dx0 = d.Measure('dx', domain=mesh0)
dx_ = d.Measure('dx', domain=mesh_)
# Give u some value
u0.assign(d.Constant(4))

# Show that the submeshes indeed are non-empty
print(d.assemble(1*dx))  # volume of full mesh = 4
print(d.assemble(1*dx0)) # 0.5
print(d.assemble(1*dx_)) # 1

# Try to assemble a function integrated over domain
print(d.assemble(u0*dx0)) # 4*0.5 = 2

# !!! THIS FAILS IN PARALLEL !!!
# for me, mpirun -np n python3 where n=[1,2,3] works fine, but n=[4,5] hangs and n=6 segfaults
print(d.assemble(u0*dx_)) # 4*1 = 4 
