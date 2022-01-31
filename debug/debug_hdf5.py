import dolfin as d
rank = d.MPI.comm_world.rank

# Load in mesh
from pathlib import Path
path    = Path('.').resolve()
subdir  = 'data'
while True:
    if path.parts[-1]=='stubs' and path.joinpath(subdir).is_dir():
        path = path.joinpath(subdir)
        break
    path = path.parent
mesh_path = str(path/'adjacent_cubes_refined.h5')
mesh = d.Mesh()
hdf5 = d.HDF5File(mesh.mpi_comm(), mesh_path, 'r')
hdf5.read(mesh, '/mesh', False)
# mesh = d.UnitCubeMesh(26, 26, 26) # something like this will work in both cases (even if it is saved/loaded from a .h5 file)
print(f"cpu {rank}: has {mesh.num_vertices()} vertices")

# define mesh functions
mf3 = d.MeshFunction("size_t", mesh, 3, 0)
mf2 = d.MeshFunction("size_t", mesh, 2, 0)
for c in d.cells(mesh):
    mf3[c] = 1+( c.midpoint().z() < 0.5 )*1
for f in d.facets(mesh):
    mf2[f] = 11*(0.5-d.DOLFIN_EPS <= f.midpoint().z() <= 0.5+d.DOLFIN_EPS)

# Mixed problem
def mixed_problem():
    # submeshes
    mesh0 = d.MeshView.create(mf3, 1); mesh1 = d.MeshView.create(mf3, 2); mesh_ = d.MeshView.create(mf2, 11)
    # build mesh mappings
    mesh_.build_mapping(mesh0); mesh_.build_mapping(mesh1)
    # function spaces
    V0 = d.FunctionSpace(mesh0, "CG", 1); V1 = d.FunctionSpace(mesh1, "CG", 1)
    # Mixed function space
    W = d.MixedFunctionSpace(V0, V1)
    # functions
    u = d.Function(W)
    un = d.Function(W)
    u0 = u.sub(0); u1 = u.sub(1)
    u0n = un.sub(0); u1n = un.sub(1)
    # test functions
    v = d.TestFunctions(W)
    v0 = v[0]; v1 = v[1]
    # measures
    dx0 = d.Measure('dx', domain=mesh0); dx1 = d.Measure('dx', domain=mesh1); dx_ = d.Measure('dx', domain=mesh_)
    # initial condition
    expression0 = d.Expression('x[0]+4', degree=1); expression1 = d.Constant(0.5)
    u0.assign(expression0); u1.assign(expression1)
    un.sub(0).assign(expression0); un.sub(1).assign(expression1)

    # define problem
    F0 = u0*v0*dx0 + d.inner(d.grad(u0), d.grad(v0))*dx0 - u0n*v0*dx0 - (d.Constant(0.01)*(u1-u0)*v0*dx_)
    F1 = u1*v1*dx1 + d.inner(d.grad(u1), d.grad(v1))*dx1 - u1n*v1*dx1 - (d.Constant(0.01)*(u0-u1)*v1*dx_)

    return u, [F0,F1]

# Try to assemble a sub-form of the mixed problem 
u, F = mixed_problem()
# assemble_mixed() requires us to separate forms by domain
from ufl.form import sub_forms_by_domain 
F00,F01 = sub_forms_by_domain(F[0])
Fvec0 = d.assemble_mixed(F00) # works in serial, MPI hangs in parallel
print(f"success! the sum of our assembled vector is {sum(Fvec0)}")

# example of a problem using no mixed assembly that works
def sub_problem():
    V = d.FunctionSpace(mesh, "CG", 1)
    u = d.Function(V)
    v = d.TestFunction(V)
    dx = d.Measure('dx', domain=mesh)

    expression = d.Expression('x[0]+4', degree=1)
    u.assign(expression)
    un = u.copy(deepcopy=True)
    F = u*v*dx + d.inner(d.grad(u), d.grad(v))*dx - un*v*dx

    return u, F

# Assembling the form in a regular problem
u, F = sub_problem()
d.assemble(F)
Fvec = d.assemble_mixed(F) # works in serial and in parallel
print(f"success! the sum of our assembled vector is {sum(Fvec)}")



