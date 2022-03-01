# mixed assembly mwe
import dolfin as d
rank = d.MPI.comm_world.rank
# comms
comm_world = d.MPI.comm_world
# split the communicator into two groups, ranks [0,1,3] and [2]
# group_013 = comm_world.group.Incl([0,1,3])
# comm_013 = comm_world.Create_group(group_013)
# group_2 = comm_world.group.Excl([0,1,3])
# comm_2 = comm_world.Create(group_2)
import stubs.config as config
config.Config().set_all_logger_levels('DEBUG')


# Create a mesh
mesh = d.UnitCubeMesh(comm_world,40,40,40)
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

# # Create submeshes using MeshView
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

# # Show that the submeshes indeed are non-empty
# print(d.assemble(1*dx))  # volume of full mesh = 4
# print(d.assemble(1*dx0)) # 0.5
# print(d.assemble(1*dx_)) # 1

# # Try to assemble a function integrated over domain
# print(d.assemble(u0*dx0)) # 4*0.5 = 2

# !!! THIS FAILS IN PARALLEL !!!
# for me, mpirun -np n python3 where n=[1,2,3] works fine, but n=[4,5] hangs and n=6 segfaults
#print(d.assemble(u0*dx_)) # 4*1 = 4 
# if rank!=2:
#     print(d.assemble_mixed(u0*dx_)) # 4*1 = 4
#print(d.assemble(u0*dx0)) # 4*1 = 4 
v0 = d.TestFunction(V0)
# import petsc4py.PETSc as p


# vec = p.Vec().createSeq(10086, comm=comm_world)
# # vec.assemble()
# vec = d.PETScVector(vec)
#vec= d.PETScVector()
# vec = p.
# dvec = d.assemble_mixed(u0*v0*dx_) # 4*1 = 4
# print(dvec.size()) #10086


#vec = d.PETScVector(comm_013)

c = list(d.cells(mesh_))[0]
print(f"{rank}: {c.mesh().topology().mapping()}")
d.assemble_mixed(u0*v0*dx_)#, tensor=vec)
#print(vec.size()) #10086



