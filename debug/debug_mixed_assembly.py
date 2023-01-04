# mixed assembly mwe
import dolfin as d

rank = d.MPI.comm_world.rank
# set log level
import logging

# Create a mesh
mesh = d.UnitCubeMesh(40, 40, 40)
z_slice = 0.5  # value of z we use to demaracate the different subvolumes
mesh.coordinates()[:, 2] *= 4  # something about partitioning leads to hanging
mesh.init()

# define mesh functions
mf3 = d.MeshFunction("size_t", mesh, 3, 0)
mf2 = d.MeshFunction("size_t", mesh, 2, 0)
for c in d.cells(mesh):
    mf3[c] = (c.midpoint().z() < z_slice + d.DOLFIN_EPS) * 1

for f in d.facets(mesh):
    mf2[f] = (z_slice - d.DOLFIN_EPS <= f.midpoint().z() <= z_slice + d.DOLFIN_EPS) * 1

# Create submeshes using MeshView
mesh0 = d.MeshView.create(mf3, 1)
mesh_ = d.MeshView.create(mf2, 1)
# build mesh mapping
mesh_.build_mapping(mesh0)
# mesh_.build_mapping(mesh)
# function space and function
V0 = d.FunctionSpace(mesh0, "CG", 1)
u0 = d.Function(V0)
v0 = d.TestFunction(V0)
# measures
dx = d.Measure("dx", domain=mesh)
dx0 = d.Measure("dx", domain=mesh0)
dx_ = d.Measure("dx", domain=mesh_)
# Give u some value
u0.assign(d.Constant(4))

# # Show that the submeshes indeed are non-empty
# print(d.assemble(1*dx))  # volume of full mesh = 4
# print(d.assemble(1*dx0)) # 0.5
# print(d.assemble(1*dx_)) # 1


d.set_log_level(10)  # highest logging
# set for others
logging.getLogger("UFL").setLevel("DEBUG")
logging.getLogger("FFC").setLevel("DEBUG")
logging.getLogger("dijitso").setLevel("DEBUG")

# bounding box debug
# mesh.bounding_box_tree()
# this fails with proc 3/4 when n=4
# print(f"{rank}: {mesh0.num_entities(0)}")

tdim = mesh0.topology().dim()
gdim = mesh0.geometric_dimension()


print(f"{rank}: {mesh0.num_entities(0)}", flush=True)

from dolfin.fem.assembling import _create_dolfin_form, _create_tensor

form = u0 * dx_
dolfin_form = _create_dolfin_form(form, None)
# Create tensor
comm = dolfin_form.mesh().mpi_comm()
tensor = _create_tensor(comm, form, dolfin_form.rank(), None, None)
# assembler.assemble(tensor, dolfin_form)

assembler = d.cpp.fem.Assembler()
from dolfin.jit.jit import ffc_jit

ufc_form = ffc_jit(form, mpi_comm=mesh0.mpi_comm())
ufc_form = d.cpp.fem.make_ufc_form(ufc_form[0])
dolfin_form.coefficients
assembler.init_global_tensor(tensor, dolfin_form)
assembler.add_values = False
assembler.finalize_tensor = False
assembler.keep_diagonal = False

new_mesh = dolfin_form.mesh()
print(dolfin_form.rank())
# if rank!=2:
assembler.assemble(tensor, dolfin_form)
# import subprocess
# output = subprocess.check_output("assembler.assemble(tensor, dolfin_form)", shell=True)
# output = subprocess.check_output("m.init(33)", shell=True)

d.MPI.barrier(d.MPI.comm_world)
# import time
# time.sleep(2)
# tree = d.cpp.geometry.BoundingBoxTree()
# mesh0.init(tdim)
# num_leaves = mesh0.num_entities(tdim)


# tree.build(mesh0, 3)
# mesh0.bounding_box_tree()


# mesh_.bounding_box_tree()


# Try to assemble a function integrated over domain
# print(d.assemble_mixed(u0*dx_, finalize_tensor=False)) # 4*0.5 = 2

# !!! THIS FAILS IN PARALLEL !!!
# for me, mpirun -np n python3 where n=[1,2,3] works fine, but n=[4,5] hangs and n=6 segfaults
# print(d.assemble_mixed(u0**3*dx_)) # 4*1 = 4
# d.solve(u0*v0*dx_ == 0, u0)
# d.assemble(u0*v0*dx0)


def rank_color(mesh, meshname):
    rank = d.MPI.comm_world.rank
    # Create rank function on mesh
    P1 = d.FiniteElement("DG", mesh.ufl_cell(), 0)
    V = d.FunctionSpace(mesh, P1)
    rank_func_mesh = d.Function(V)
    rank_func_mesh.assign(d.Constant(-1))  # indicates cells not owned by a process
    rank_func_mesh.rename("rank on mesh", "rank on mesh")
    ndofs = rank_func_mesh.vector().vec().getArray().size
    for i in range(ndofs):
        rank_func_mesh.vector().vec().getArray(readonly=0)[i] = rank + 1
    rank_func_mesh.vector().update_ghost_values()
    rank_func_mesh.vector().apply("insert")

    # Save
    nprocs = d.MPI.comm_world.size
    with d.XDMFFile(d.MPI.comm_world, f"rank_on_{meshname}_{nprocs}.xdmf") as file:
        file.write(rank_func_mesh, d.XDMFFile.Encoding.HDF5)


# rank_color(mesh0, 'mesh0')
# rank_color(mesh, 'mesh')
# # print(d.MPI.comm_world.size)
