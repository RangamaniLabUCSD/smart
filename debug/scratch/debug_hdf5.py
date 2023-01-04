import dolfin as d
import numpy as np

rank = d.MPI.comm_world.rank
comm = d.MPI.comm_world

# Load in mesh
from pathlib import Path

path = Path(".").resolve()
subdir = "data"
while True:
    if path.parts[-1] == "stubs" and path.joinpath(subdir).is_dir():
        path = path.joinpath(subdir)
        break
    path = path.parent
# mesh_path = str(path/'adjacent_cubes_refined.h5')
# mesh = d.Mesh(comm)
# hdf5 = d.HDF5File(mesh.mpi_comm(), mesh_path, 'r')
# hdf5.read(mesh, '/mesh', False)
# mesh.init()
# #mesh = d.UnitCubeMesh(26, 26, 26) # something like this will work in both cases (even if it is saved/loaded from a .h5 file)
# #print(f"cpu {rank}: has {mesh.num_vertices()} vertices")


# # define mesh functions
# mf3 = d.MeshFunction("size_t", mesh, 3, 0)
# mf2 = d.MeshFunction("size_t", mesh, 2, 0)
# for c in d.cells(mesh):
#     mf3[c] = 1+( c.midpoint().z() < 0.0 )*1
# for f in d.facets(mesh):
#     mf2[f] = 11*(0.0-d.DOLFIN_EPS <= f.midpoint().z() <= 0.0+d.DOLFIN_EPS)
#     # mf2[f] = 13*(1.0-d.DOLFIN_EPS <= f.midpoint().z() <= 1.0+d.DOLFIN_EPS)

# Mixed problem
def mixed_problem():
    # submeshes
    mesh0 = d.MeshView.create(mf3, 1)
    mesh1 = d.MeshView.create(mf3, 2)
    mesh_ = d.MeshView.create(mf2, 11)
    for submesh in [mesh0, mesh1, mesh_]:
        submesh.init()
    # build mesh mappings
    mesh_.build_mapping(mesh0)
    mesh_.build_mapping(mesh1)
    # function spaces
    V0 = d.FunctionSpace(mesh0, "CG", 1)
    V1 = d.FunctionSpace(mesh1, "CG", 1)
    # Mixed function space
    W = d.MixedFunctionSpace(V0, V1)
    # functions
    u = d.Function(W)
    un = d.Function(W)
    u0 = u.sub(0)
    u1 = u.sub(1)
    u0n = un.sub(0)
    u1n = un.sub(1)
    # test functions
    v = d.TestFunctions(W)
    v0 = v[0]
    v1 = v[1]
    # measures
    dx0 = d.Measure("dx", domain=mesh0)
    dx1 = d.Measure("dx", domain=mesh1)
    dx_ = d.Measure("dx", domain=mesh_)
    ds0 = d.Measure("ds", domain=mesh0)
    ds1 = d.Measure("ds", domain=mesh1)
    # initial condition
    expression0 = d.Expression("x[0]+4", degree=1)
    expression1 = d.Constant(0.5)
    u0.assign(expression0)
    u1.assign(expression1)
    un.sub(0).assign(expression0)
    un.sub(1).assign(expression1)

    # sanity check
    n = d.FacetNormal(mesh0)
    n2 = d.Constant((0, 0, 1))
    print(f"facet normal dot product: {sum(d.assemble(d.dot(n,n2)*v0*ds0))}")
    # print(d.assemble_mixed(u0*dx_))
    print(d.assemble(u0 * dx_))

    # define problem
    # F0 = u0*v0*dx0 + d.inner(d.grad(u0), d.grad(v0))*dx0 - u0n*v0*dx0 - (d.Constant(0.01)*(u1-u0)*v0*dx_)
    # F1 = u1*v1*dx1 + d.inner(d.grad(u1), d.grad(v1))*dx1 - u1n*v1*dx1 - (d.Constant(0.01)*(u0-u1)*v1*dx_)
    F0 = (
        u0 * v0 * dx0
        + d.inner(d.grad(u0), d.grad(v0)) * dx0
        - u0n * v0 * dx0
        - (d.Constant(0.01) * u0 * v0 * dx_)
    )
    F1 = (
        u1 * v1 * dx1
        + d.inner(d.grad(u1), d.grad(v1)) * dx1
        - u1n * v1 * dx1
        - (d.Constant(0.01) * v1 * dx_)
    )

    return u, [F0, F1], [mesh0, mesh1, mesh_]


# # Try to assemble a sub-form of the mixed problem
# u, F, meshes = mixed_problem()
# mesh0, mesh1, mesh_ = meshes
# # assemble_mixed() requires us to separate forms by domain
# from ufl.form import sub_forms_by_domain
# F00,F01 = sub_forms_by_domain(F[0])
# tensor = d.PETScVector()
# d.assemble_mixed(F00, tensor=tensor) # works in serial, MPI hangs in parallel
# print(f"success! the sum of our assembled vector is {sum(Fvec0)}")

# # example of a problem using no mixed assembly that works
# def sub_problem():
#     V = d.FunctionSpace(mesh, "CG", 1)
#     u = d.Function(V)
#     v = d.TestFunction(V)
#     dx = d.Measure('dx', domain=mesh)

#     expression = d.Expression('x[0]+4', degree=1)
#     u.assign(expression)
#     un = u.copy(deepcopy=True)
#     F = u*v*dx + d.inner(d.grad(u), d.grad(v))*dx - un*v*dx

#     return u, F

# # Assembling the form in a regular problem
# u, F = sub_problem()
# d.assemble(F)
# Fvec = d.assemble_mixed(F) # works in serial and in parallel
# print(f"success! the sum of our assembled vector is {sum(Fvec)}")


# d.parameters
# d.parameters["mesh_partitioner"] = "ParMETIS"
# d.parameters["ghost_mode"] = "shared_cell"
# Simpler example
mesh_type = 2
if mesh_type == 1:
    # mesh_ has 1089 vertices, 2048 triangles
    mesh_path = str(path / "adjacent_cubes_refined2.h5")
    mesh = d.Mesh()
    hdf5 = d.HDF5File(mesh.mpi_comm(), mesh_path, "r")
    hdf5.read(mesh, "/mesh", False)
    midpoint = 0.0
elif mesh_type == 2:
    mesh = d.UnitCubeMesh(40, 40, 40)
    midpoint = 0.5
    mesh.coordinates()[:, 2] *= 4
elif mesh_type == 3:
    mesh = d.UnitCubeMesh(comm, 16, 16, 16)
    midpoint = 0.5
elif mesh_type == 4:
    mesh_path = str(path / "unit_cube_mesh_16.h5")
    mesh = d.Mesh(comm)
    hdf5 = d.HDF5File(mesh.mpi_comm(), mesh_path, "r")
    hdf5.read(mesh, "/mesh", False)
    midpoint = 0.5
elif mesh_type == 5:
    mesh_path = str(path / "unit_cube_mesh_16_xmltoh5.h5")
    mesh = d.Mesh(comm)
    hdf5 = d.HDF5File(mesh.mpi_comm(), mesh_path, "r")
    hdf5.read(mesh, "/mesh", False)
    midpoint = 0.75


mesh.init()


def print_mpi_mesh_info(mesh):
    shared_vertices = np.fromiter(
        mesh.topology().shared_entities(0).keys(),
        dtype="uintc",
    )
    shared_cells = mesh.topology().shared_entities(mesh.topology().dim())
    shared_facets = mesh.topology().shared_entities(1)
    num_regular_vertices = mesh.topology().ghost_offset(0)
    ghost_vertices = np.arange(num_regular_vertices, mesh.topology().size(0))
    # print(f"cpu {rank}: shared_vertices = {shared_vertices}")
    print(
        f"cpu {rank}: number of vertices/shared_vertices = {mesh.num_vertices()}, {len(shared_vertices)}",
    )
    print(
        f"cpu {rank}: number of cells/shared_cells = {mesh.num_cells()}, {len(shared_cells)}",
    )
    # print(f"cpu {rank}: number of facets/shared_facets = {mesh.num_facets()}, {len(shared_facets)}")
    # print(f"cpu {rank}: number of shared_vertices = {len(shared_vertices)}")
    # print(f"cpu {rank}: number of shared_cells = {len(shared_cells)}")

    # print(f"cpu {rank}: shared_facets = {shared_facets}")
    # print(f"cpu {rank}: ghost_vertices = {ghost_vertices}")


# define mesh functions
mf3 = d.MeshFunction("size_t", mesh, 3, 0)
mf2 = d.MeshFunction("size_t", mesh, 2, 0)
for c in d.cells(mesh):
    mf3[c] = (c.midpoint().z() < midpoint + d.DOLFIN_EPS) * 1
for f in d.facets(mesh):
    mf2[f] = 11 * (
        midpoint - d.DOLFIN_EPS <= f.midpoint().z() <= midpoint + d.DOLFIN_EPS
    )

mesh0 = d.MeshView.create(mf3, 1)
mesh_ = d.MeshView.create(mf2, 11)
# build mesh mapping
mesh_.build_mapping(mesh0)
# function space
V0 = d.FunctionSpace(mesh0, "CG", 1)
DG0 = d.FunctionSpace(mesh0, "DG", 0)
# function
u0 = d.Function(V0)
# measure
dx = d.Measure("dx", domain=mesh)
dx0 = d.Measure("dx", domain=mesh0)
dx_ = d.Measure("dx", domain=mesh_)
# initial condition
u0.assign(d.Constant(4))

# print_mpi_mesh_info(mesh0)
print_mpi_mesh_info(mesh)

# print(f"cpu {rank}: length of vertex map = {len(mapping.vertex_map())}")
# print(f"cpu {rank}: length of cell map = {len(mapping.cell_map())}")
# print('\n')

# # gather cell mappings through MPI
# from mpi4py import MPI
# mapping = mesh_.topology().mapping()[0]
# cell_map = np.array(mapping.cell_map(), dtype=int)
# vertex_map = np.array(mapping.vertex_map(), dtype=int)

# sendbuf = np.array(cell_map)
# # Collect local array sizes using the high-level mpi4py gather
# sendcounts = np.array(comm.gather(len(sendbuf), 0))
# if rank == 0:
#     print("sendcounts: {}, total: {}".format(sendcounts, sum(sendcounts)))
#     recvbuf = np.empty(sum(sendcounts), dtype=int)
# else:
#     recvbuf = None

# comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=0)

# if rank == 0:
#     print(len(recvbuf))
#     print(len(np.unique(recvbuf)))
#     # np.savetxt('6proc.csv', recvbuf, delimiter=',')


# Form
from ufl.form import sub_forms_by_domain
from dolfin.fem.assembling import _create_dolfin_form, _create_tensor

form = u0 * dx_
# print(f"cpu {rank}: num vertices on form = {dolfin_form.mesh().num_vertices()}")

print(f"cpu {rank}: ownership range = {u0.function_space().dofmap().ownership_range()}")
print(f"cpu {rank}: ownership range (DG0) = {DG0.dofmap().ownership_range()}")
m_map = mesh_.topology().mapping()[0].cell_map()
if len(m_map) > 0:
    print(f"cpu {rank}: vertex meshview = {min(m_map)}, {max(m_map)}")

print("\n")

# tensor = d.Scalar(mesh.mpi_comm())
# d.assemble_mixed(u0*dx_, tensor=tensor)
# d.assemble(form)
# Create dolfin Form object referencing all data needed by assembler
# if isinstance(form, d.cpp.fem.Form):
#     dolfin_forms = [form]
# else:
#     dolfin_forms = []
#     for subform in sub_forms_by_domain(form):
#         dolfin_forms.append(d.fem.assembling._create_dolfin_form(subform, None))

# Create dolfin Form object
if isinstance(form, d.cpp.fem.Form):
    dolfin_form = form
else:
    dolfin_form = _create_dolfin_form(form, None)

# Create tensor
comm = dolfin_form.mesh().mpi_comm()
tensor = _create_tensor(comm, form, dolfin_form.rank(), None, None)

# d.assemble_mixed(dolfin_form, tensor=tensor)
# d.assemble(u0*dx_)
d.assemble(u0 * dx0)

# # Create C++ assembler
assembler = d.cpp.fem.Assembler()
# # Set assembler options
assembler.add_values = False
assembler.finalize_tensor = True
assembler.keep_diagonal = False

# # Call C++ assemble
# assembler.assemble(dolfin_form)
# assembler.assemble(tensor, dolfin_form)

# # Convert to float for scalars
# if dolfin_form.rank() == 0:
#     tensor = tensor.get_scalar_value()


# print('assembled form!')
# print(tensor)

# d.assemble(form)
# print('assembled form!')
# print(d.assemble_mixed(u0*dx_))

# if rank==0:
#     print(d.assemble(1*dx))
# print(d.assemble(1*dx0))
# print(d.assemble(1*dx_))

# #print(d.assemble_mixed(u0*dx_))
# print(d.assemble(u0*dx0))
# print(d.assemble(u0*dx_))


def rank_color(submesh):
    # # Create rank function on mesh
    # P1 = d.FiniteElement("DG", mesh.ufl_cell(), 0)
    # V = d.FunctionSpace(mesh, P1)
    # rank_func_mesh = d.Function(V)
    # rank_func_mesh.rename("rank on mesh","rank on mesh")
    # ndofs = rank_func_mesh.vector().vec().getArray().size
    # for i in range(ndofs):
    #     rank_func_mesh.vector().vec().getArray(readonly=0)[i] = rank+1
    # rank_func_mesh.vector().update_ghost_values()
    # #rank_func_mesh.vector().apply("insert")

    # Create rank function on submesh
    P1sub = d.FiniteElement("DG", submesh.ufl_cell(), 0)
    Vsub = d.FunctionSpace(submesh, P1sub)
    rank_func_submesh = d.Function(Vsub)
    rank_func_submesh.rename("rank on submesh", "rank on submesh")
    subndofs = rank_func_submesh.vector().vec().getArray().size
    for i in range(subndofs):
        rank_func_submesh.vector().vec().getArray(readonly=0)[i] = rank + 1
    rank_func_submesh.vector().update_ghost_values()
    # rank_func_submesh.vector().apply("insert")

    # # Save
    # with d.XDMFFile(d.MPI.comm_world, "rank_on_mesh.xdmf") as file:
    #     file.write(rank_func_mesh,d.XDMFFile.Encoding.HDF5)

    with d.XDMFFile(d.MPI.comm_world, "rank_on_submesh_4_adj.xdmf") as file:
        file.write(rank_func_submesh, d.XDMFFile.Encoding.HDF5)


rank_color(mesh)

# if rank==0:
#     np.savetxt('2proc.csv', recvbuf, delimiter=',')


# # on serial
# a = recvbuf
# b = np.genfromtxt('2proc.csv', delimiter=',')
# b = b.astype(int)
# b = np.genfromtxt('3proc.csv', delimiter=',')
# b = b.astype(int)
# b = np.genfromtxt('6proc.csv', delimiter=',')
# b = b.astype(int)
# m_map = mesh_.topology().mapping()[0].cell_map()


# # Create function on mesh (serial)
# DG0 = d.FiniteElement("DG", mesh_.ufl_cell(), 0)
# V = d.FunctionSpace(mesh_, DG0)
# func_mesh = d.Function(V)
# func_mesh.rename("func on mesh","func on mesh")
# #for i in range(ndofs):
# for i in a:
#     idx = m_map.index(i)
#     func_mesh.vector().vec().getArray(readonly=0)[idx] = 1

# func_mesh.vector().update_ghost_values()
# #rank_func_mesh.vector().apply("insert")

# # Create function on mesh (parallel)
# func_mesh_p = d.Function(V)
# func_mesh_p.rename("func on mesh (parallel)","func on mesh (parallel)")
# #for i in range(subndofs):
# for i in b:
#     try:
#         idx = m_map.index(i)
#         func_mesh_p.vector().vec().getArray(readonly=0)[idx] = 1
#     except:
#         pass

# func_mesh_p.vector().update_ghost_values()
# #rank_func_submesh.vector().apply("insert")

# # Save
# with d.XDMFFile(d.MPI.comm_world, "func_mesh.xdmf") as file:
#     file.write(func_mesh,d.XDMFFile.Encoding.HDF5)

# with d.XDMFFile(d.MPI.comm_world, "func_mesh_6.xdmf") as file:
#     file.write(func_mesh_p,d.XDMFFile.Encoding.HDF5)


# mesh = d.UnitCubeMesh(16,16,16)
# d.File('unit_cube_mesh_16.xml') << mesh

# readmesh = d.Mesh('unit_cube_mesh_16.xml')

# # write
# hdf5 = d.HDF5File(readmesh.mpi_comm(), 'unit_cube_mesh_16_xmltoh5.h5', 'w')
# hdf5.write(readmesh, '/mesh')
# hdf5.close()
