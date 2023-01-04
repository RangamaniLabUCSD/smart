# testing petsc snes
# https://fenicsproject.discourse.group/t/using-petsc4py-petsc-snes-directly/2368/12
# ====================
# stubs
# ===================
from copy import deepcopy

import dolfin as d

import stubs

rank = d.MPI.comm_world.rank
print(f"rank = {rank}")

from pathlib import Path

path = Path(".").resolve()
subdir = "data"
while True:
    if path.parts[-1] == "stubs" and path.joinpath(subdir).is_dir():
        path = path.joinpath(subdir)
        break
    path = path.parent

# adjacent_cubes_mesh = stubs.mesh.ParentMesh(str(path / 'adjacent_cubes_refined.h5'), 'hdf5')

# dolfin adjacent_cubes_mesh
print("loading mesh")
# mesh_path = str(path/'adjacent_cubes_refined2.h5')
# mesh_path = str(path/'unit_cube_mesh_16.h5')
mesh = d.UnitCubeMesh(24, 24, 24)
# mesh0 = d.Mesh()
# hdf5 = d.HDF5File(mesh0.mpi_comm(), mesh_path, 'r')
# hdf5.read(mesh0, '/mesh', False)
# # mesh = d.Mesh(str(path/'adjacent_cubes.xml'))
# print(mesh0.num_vertices())
# mesh=mesh0
# mesh = d.refine(mesh0)
print(mesh.num_vertices())

# hdf5 = d.HDF5File(mesh.mpi_comm(), str(path/'unit_cube_mesh_16.h5'), 'w')
# hdf5.write(mesh, '/mesh')
# hdf5.close()

# self.dolfin_mesh.init()

mf3_ = d.MeshFunction("size_t", mesh, 3, 0)
mf2_ = d.MeshFunction("size_t", mesh, 2, 0)
for c in d.cells(mesh):
    mf3_[c] = 11 + (c.midpoint().z() < 0.5) * 1

for f in d.facets(mesh):
    mf2_[f] = 4 * (0.5 - d.DOLFIN_EPS <= f.midpoint().z() <= 0.5 + d.DOLFIN_EPS)


# import stubs_model
# model = stubs_model.make_model(refined_mesh=True)

# # init model
# model._init_1()
# model._init_2()
# model._init_3()
# model._init_4()
# model._init_5_1_reactions_to_fluxes()
# model._init_5_2_create_variational_forms()
# model._init_5_3_create_variational_problem()

# ====================
# petsc snes
# ===================
import petsc4py, sys

petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
import ufl
from ufl.form import sub_forms_by_domain

# trying petscnestmatrix to init vectors of appropriate size
# model.Jlist = [[J00(omega0), J00(omega1)], [J01(omega0)], etc]
def get_nestmat_from_Jlist(Jlist):
    # For some reason assemble_mixed() doesn't work on forms with multiple integration domains, so the Jlist
    # that is fed to d.MixedNonlinearVariationalSolver is in a form like
    # [[J0(Omega_0), J0(Omega_1)], ..., [Jn(Omega_n)]]
    new_Jlist = list()
    for Jsublist in Jlist:
        Jsub = [d.assemble_mixed(J) for J in Jsublist]
        J = Jsub[0]
        if len(Jsub) > 1:
            for i in range(1, len(Jsub)):
                J += Jsub[i]

        new_Jlist.append(J)

    return new_Jlist


class mySNESProblem:
    def __init__(self, F, u, J):  # Jlist):
        self.L = F
        self.a = J
        # because this is from a mixedfunctionspace
        self.u = u
        # save sparsity patterns of block matrices
        # self.tensors = [[None]*len(Jij_list) for Jij_list in self.Jlist]

    def F(self, snes, x, F):
        x = d.PETScVector(x)  # convert petsc.Vec -> d.PETScVector
        F = d.PETScVector(F)
        x.vec().copy(self.u.vector().vec())
        self.u.vector().apply("")
        d.assemble(self.L, tensor=F)

    def J(self, snes, x, J, P):
        x = d.PETScVector(x)  # convert petsc.Vec -> d.PETScVector
        # convert from petsc4py type to dolfin wrapper
        # J = d.PETScNestMatrix(J)
        J = d.PETScMatrix(J)
        # copy values of x -> u
        x.vec().copy(self.u.vector().vec())
        self.u.vector().apply("")
        d.assemble(self.a, tensor=J)

        # Alist = list()
        # Aij_list = [[None]*len(Jij_list) for Jij_list in self.Jlist]
        # for ij, Jij_list in enumerate(self.Jlist):
        #     # Jijk == dFi/duj(Omega_k)
        #     for k, Jijk in enumerate(Jij_list):
        #         # if we have the sparsity pattern re-use it, if not save it for next time
        #         Aij_list[ij].append(d.assemble_mixed(Jijk, tensor=self.tensors[ij][k]))
        #         if self.tensors[ij][k] is None:
        #             self.tensors[ij][k] = Aij_list[ij]

        #     # sum the matrices
        #     Aij = Aij_list[ij][0]
        #     for k in range(1, len(Aij_list[ij])):
        #         Aij += Aij_list[ij][k]

        #     Alist.append(Aij)

        # J = d.PETScNestMatrix(Alist)


class mySNESProblem_nest:
    def __init__(self, u, Fforms, Jforms):
        self.u = u
        self.Fforms = Fforms
        self.Jforms = Jforms

        # List of lists (lists partitioned by integration domains)
        assert isinstance(Fforms, list)
        assert isinstance(Fforms[0], list)
        assert isinstance(Jforms, list)
        assert isinstance(Jforms[0], list)

        self.dim = len(Fforms)
        assert len(Jforms) == self.dim**2

        # save sparsity patterns of block matrices
        self.tensors = [[None] * len(Jij_list) for Jij_list in self.Jforms]

    def initialize_petsc_matnest(self):
        assert isinstance(self.Jforms[0][0], (ufl.Form, d.Form))
        # dim = int(np.sqrt(len(Forms)))
        # assert dim**2 == len(Forms)
        dim = self.dim

        # Jdpetsc = [[None]*dim]*dim
        Jpetsc = []
        for i in range(dim):
            for j in range(dim):
                ij = i * dim + j
                # Jdpetsc[i][j] = d.PETScMatrix()
                Jsum = d.as_backend_type(
                    d.assemble_mixed(self.Jforms[ij][0]),
                )  # , tensor=Jdpetsc[i][j])
                for k in range(1, len(self.Jforms[ij])):
                    Jsum += d.as_backend_type(
                        d.assemble_mixed(self.Jforms[ij][k]),
                    )  # , tensor=Jdpetsc[i][j])

                Jpetsc.append(Jsum)  # Jdpetsc[i][j].mat()
                # Jpetsc[i][j] = Jsum.mat()#Jdpetsc[i][j].mat()
                # Jdpetsc[i][j] = Jsum#Jdpetsc[i][j].mat()
                # print(f"init i={i}, j={j}: size={Jpetsc[ij].size}")

        # Jdpetsc_nest =  d.PETScNestMatrix([Jdpetsc[0][0], Jdpetsc[0][1], Jdpetsc[1][0], Jdpetsc[1][1]])
        # Jpetsc_nest =  PETSc.Mat().createNest(Jpetsc)
        Jpetsc_nest = d.PETScNestMatrix(Jpetsc).mat()

        return Jpetsc_nest

    def initialize_petsc_vecnest(self):
        assert isinstance(self.Fforms[0][0], (ufl.Form, d.Form))
        # dim = len(Forms)
        dim = self.dim

        # Fdpetsc = [None]*dim
        Fpetsc = []
        for j in range(dim):
            # Fdpetsc[j] = d.PETScVector()
            Fsum = d.as_backend_type(
                d.assemble_mixed(self.Fforms[j][0]),
            )  # , tensor=Fdpetsc[j])
            for k in range(1, len(self.Fforms[j])):
                Fsum += d.as_backend_type(
                    d.assemble_mixed(self.Fforms[j][k]),
                )  # , tensor=Fdpetsc[j])

            Fpetsc.append(Fsum.vec())

        Fpetsc_nest = PETSc.Vec().createNest(Fpetsc)
        # # also return a zero vector for the block structure
        # upetsc = Fpetsc_nest.copy()
        # upetsc.zeroEntries()

        return Fpetsc_nest  # , upetsc

    def assemble_Jnest(self, Jnest):
        """Assemble Jacobian nest matrix

        Parameters
        ----------
        Jnest : petsc4py.Mat
            PETSc nest matrix representing the Jacobian

        Jmats are created using assemble_mixed(Jform) and are dolfin.PETScMatrix types
        """
        dim = self.dim
        # Jmats = [[None]*dim]*dim #list of two lists
        Jmats = []
        # Get the petsc sub matrices, convert to dolfin wrapper, assemble forms using dolfin wrapper as tensor
        # for ij, Jij_forms in enumerate(self.Jforms):
        for i in range(dim):
            for j in range(dim):
                ij = i * dim + j
                Jij_petsc = Jnest.getNestSubMatrix(i, j)
                Jmats.append([])
                # Jijk == dFi/duj(Omega_k)
                for k, Jijk_form in enumerate(self.Jforms[ij]):
                    # if we have the sparsity pattern re-use it, if not save it for next time
                    # Jmats[ij].append(d.assemble_mixed(Jijk_form, tensor=self.tensors[ij][k]))
                    Jmats[ij].append(
                        d.assemble_mixed(Jijk_form, tensor=d.PETScMatrix()),
                    )
                    if self.tensors[ij][k] is None:
                        self.tensors[ij][k] = Jmats[ij]

                # sum the matrices
                Jij_petsc.zeroEntries()
                # for k in range(1, len(Jmats[ij])):
                for Jmat in Jmats[ij]:
                    Jij_petsc.axpy(1, d.as_backend_type(Jmat).mat())
                # try assembling here in loop
                Jij_petsc.assemble()

        # for i in range(dim):
        #     for j in range(dim):
        #         idx = i*dim+j
        #         Jmats[i][j] = Jnest.getNestSubMatrix(i,j)
        #         # set block size?
        #         d.assemble(self.Jforms[idx][0], tensor=d.PETScMatrix(Jmats[i][j]))
        # Now use petsc assemble()
        # for i in range(dim):
        #     for j in range(dim):
        #         for k in range(len(Jmats[ij][k]))
        #             Jmats[i][j].assemble()
        Jnest.assemble()

    def assemble_Fnest(self, Fnest):
        dim = self.dim
        Fi_petsc = Fnest.getNestSubVecs()
        Fvecs = []
        for j in range(dim):
            Fvecs.append([])
            for k in range(len(self.Fforms[j])):
                Fvecs[j].append(
                    d.as_backend_type(d.assemble_mixed(self.Fforms[j][k])),
                )  # , tensor=d.PETScVector(Fvecs[idx]))
            # sum the vectors
            Fi_petsc[j].zeroEntries()
            for k in range(len(self.Fforms[j])):
                Fi_petsc[j].axpy(1, Fvecs[j][k].vec())

        for j in range(dim):
            Fi_petsc[j].assemble()
        # Fvecs = Fnest.getNestSubVecs()
        # for idx in range(dim):
        #     d.assemble(self.Fforms[idx][0], tensor=d.PETScVector(Fvecs[idx]))
        # for idx in range(dim):
        #     Fvecs[idx].assemble()

    def copy_u(self, xnest):
        uvecs = xnest.getNestSubVecs()
        duvecs = [None] * self.dim
        for idx, uvec in enumerate(uvecs):
            duvecs[idx] = d.PETScVector(uvec)  # convert petsc.Vec -> d.PETScVector
            duvecs[idx].vec().copy(self.u.sub(idx).vector().vec())
            self.u.sub(idx).vector().apply("")

    def F(self, snes, x, Fnest):
        self.copy_u(x)
        self.assemble_Fnest(Fnest)
        # F  = d.PETScVector(F)
        # d.assemble(self.Fforms, tensor=F)

    def J(self, snes, x, Jnest, P):
        self.copy_u(x)
        self.assemble_Jnest(Jnest)
        # d.assemble(self.a, tensor=d.PETScMatrix(J00))


# ====================================
# sub problem (pure diffusion)
# ====================================
def sub_problem(use_stubs=False):
    if use_stubs:
        u1 = model.u["u"].sub(1)
        v1 = model.v[1]
        dx1 = model.child_meshes["er_vol"].dx
    else:
        # sub problem not using stubs
        mesh = d.UnitCubeMesh(10, 10, 10)
        V = d.FunctionSpace(mesh, "CG", 1)
        u1 = d.Function(V)
        v1 = d.TestFunction(V)
        dx1 = d.Measure("dx")

    expression = d.Expression("x[0]+2", degree=1)
    u1.assign(expression)
    u1n = u1.copy(deepcopy=True)
    print(f"init: u1 min = {u1.vector()[:].min()}, u1 max = {u1.vector()[:].max()}")
    F = u1 * v1 * dx1 + d.inner(d.grad(u1), d.grad(v1)) * dx1 - u1n * v1 * dx1
    J = d.derivative(F, u1)

    return u1, F, J


def mixed_problem():
    # mesh = adjacent_cubes_mesh.dolfin_mesh
    # mesh = mesh
    mf3 = mf3_
    mf2 = mf2_
    # mesh functions
    # mf3 = adjacent_cubes_mesh._get_mesh_function(3); mf2 = adjacent_cubes_mesh._get_mesh_function(2)
    # submeshes
    mesh1 = d.MeshView.create(mf3, 11)
    mesh2 = d.MeshView.create(mf3, 12)
    mesh_ = d.MeshView.create(mf2, 4)
    # build mesh mappings
    mesh_.build_mapping(mesh1)
    mesh_.build_mapping(mesh2)
    # function spaces
    V1 = d.FunctionSpace(mesh1, "CG", 1)
    V2 = d.FunctionSpace(mesh2, "CG", 1)
    # Mixed function space
    W = d.MixedFunctionSpace(V1, V2)
    # functions
    u = d.Function(W)
    un = d.Function(W)
    u1 = u.sub(0)
    u2 = u.sub(1)
    u1n = un.sub(0)
    u2n = un.sub(1)
    # test functions
    v = d.TestFunctions(W)
    v1 = v[0]
    v2 = v[1]
    # measures
    dx1 = d.Measure("dx", domain=mesh1)
    dx2 = d.Measure("dx", domain=mesh2)
    dx_ = d.Measure("dx", domain=mesh_)
    # un
    expression1 = d.Expression("x[0]+4", degree=1)
    expression2 = d.Constant(0.5)
    u1.assign(expression1)
    u2.assign(expression2)
    un.sub(0).assign(expression1)
    un.sub(1).assign(expression2)
    try:
        print(
            f"init: u1 min = {u1.vector().get_local()[:].min()}, u1 max = {u1.vector().get_local()[:].max()}",
        )
        print(
            f"init: u2 min = {u2.vector().get_local()[:].min()}, u2 max = {u2.vector().get_local()[:].max()}",
        )
    except:
        print(f"proc {rank} does not have u1/u2 min/max")
    # define problem
    F1 = (
        u1 * v1 * dx1
        + d.inner(d.grad(u1), d.grad(v1)) * dx1
        - u1n * v1 * dx1
        - (d.Constant(0.01) * (u2 - u1) * v1 * dx_)
    )  # (-(u1-u2)*v1*dx_)
    F2 = (
        u2 * v2 * dx2
        + d.inner(d.grad(u2), d.grad(v2)) * dx2
        - u2n * v2 * dx2
        - (d.Constant(0.01) * (u1 - u2) * v2 * dx_)
    )  # ((u1-u2)*v2*dx_)

    J11 = d.derivative(F1, u1)
    J12 = d.derivative(F1, u2)
    J21 = d.derivative(F2, u1)
    J22 = d.derivative(F2, u2)

    Fblocks, Jblocks = stubs.model.Model.get_block_system(F1 + F2, u._functions)

    return u, un, Fblocks, Jblocks, [F1, F2], [J11, J12, J21, J22]


# print(f"\n\nMixed problem =============")
u, un, Fblocks, Jblocks, F, J = mixed_problem()
Fsum = F[0] + F[1]

# print(f"solve: u1 min = {u.sub(0).vector()[:].min()}, u1 max = {u.sub(0).vector()[:].max()}")
# print(f"solve: u2 min = {u.sub(1).vector()[:].min()}, u2 max = {u.sub(1).vector()[:].max()}")
# d.solve(Fsum==0, u)
# print("after solve")
# print(f"solve: u1 min = {u.sub(0).vector()[:].min()}, u1 max = {u.sub(0).vector()[:].max()}")
# print(f"solve: u2 min = {u.sub(1).vector()[:].min()}, u2 max = {u.sub(1).vector()[:].max()}")
# print(f"Mixed problem =============\n\n")


def snes_solve(snesproblem, Fvec, Jmat):
    snes = PETSc.SNES().create(d.MPI.comm_world)
    snes.setFunction(snesproblem.F, Fvec)
    snes.setJacobian(snesproblem.J, Jmat)
    snes.solve(None, snesproblem.u.vector().vec())
    # u1 = snesproblem.u.sub(0)
    # print(f"solve: u1 min = {u1.vector()[:].min()}, u1 max = {u1.vector()[:].max()}")
    # print(np.isclose(u1.vector()[:].max() - u1.vector()[:].min(), .4842))
    # if u1.vector()[:].min() > 2.1:
    #     print("reasonable solution")


def snes_solve_mixed(snesproblem, Fvec, Jmat, upetsc):
    snes = PETSc.SNES().create(d.MPI.comm_world)
    snes.setFunction(snesproblem.F, Fvec)
    snes.setJacobian(snesproblem.J, Jmat)
    # snes.solve(None, snesproblem.u.vector().vec())
    snes.solve(None, upetsc)
    # u1 = snesproblem.u.sub(0)
    # print(f"solve: u1 min = {u1.vector()[:].min()}, u1 max = {u1.vector()[:].max()}")
    # print(np.isclose(u1.vector()[:].max() - u1.vector()[:].min(), .4842))
    # if u1.vector()[:].min() > 2.1:
    #     print("reasonable solution")


# u1, F, J = sub_problem(use_stubs=False)
# snesproblem = mySNESProblem(F, u1, J)
# Fvec = d.PETScVector()
# Jmat = d.PETScMatrix()
# snes_solve(snesproblem.F, snesproblem.J, Fvec.vec(), Jmat.mat())

# ====================================
# Nest on mixed problem
# ====================================


# print(f"\n Nest on sub problem")
# u1, F, J = sub_problem(use_stubs=False)
# snesproblem = mySNESProblem_nest(F, u1, [J])
# Fvec = d.PETScVector()
# Jmat = d.PETScNestMatrix()
# J
# Jdpetsc = d.PETScMatrix()
# d.assemble(J, tensor=Jdpetsc)
# Jpetsc = Jdpetsc.mat()
# Jpetsc_nest = PETSc.Mat().createNest([[Jpetsc, Jpetsc], [Jpetsc, Jpetsc]])


print(f"\n Nest on mixed problem")
u, un, Fblocks, Jblocks, F, J = mixed_problem()
snesproblem = mySNESProblem_nest(u, Fblocks, Jblocks)
# create nest matrix
d.assemble_mixed(Jblocks[0][0], tensor=d.PETScMatrix())
Jpetsc_nest = snesproblem.initialize_petsc_matnest()
Fpetsc_nest = snesproblem.initialize_petsc_vecnest()
# upetsc = PETSc.Vec().createNest([u.sub(0).vector().vec().copy(), u.sub(1).vector().vec().copy()])
upetsc = PETSc.Vec().createNest([u.vector().vec().copy() for u in u._functions])

print("before solve")
try:
    print(
        f"solve: u1 min = {u.sub(0).vector().get_local()[:].min()}, u1 max = {u.sub(0).vector().get_local()[:].max()}",
    )
    print(
        f"solve: u2 min = {u.sub(1).vector().get_local()[:].min()}, u2 max = {u.sub(1).vector().get_local()[:].max()}",
    )
except:
    print(f"proc {rank} does not have u1/u2 min/max")

snes_solve_mixed(snesproblem, Fpetsc_nest, Jpetsc_nest, upetsc)

# snes.solve(None, upetsc)

print("after solve")
try:
    print(
        f"solve: u1 min = {u.sub(0).vector().get_local()[:].min()}, u1 max = {u.sub(0).vector().get_local()[:].max()}",
    )
    print(
        f"solve: u2 min = {u.sub(1).vector().get_local()[:].min()}, u2 max = {u.sub(1).vector().get_local()[:].max()}",
    )
except:
    print(f"proc {rank} does not have u1/u2 min/max")


# # J00, J01 = sub_forms_by_domain(J[0])
# J000, J001 = Jblocks[0]
# M000 = d.as_backend_type(d.assemble_mixed(J000))#;
# M000 += d.as_backend_type(d.assemble_mixed(J001))
# #M000.axpy(1, M001, False)
# J01 = Jblocks[1][0]
# M01 = d.as_backend_type(d.assemble_mixed(J01))
# J10 = Jblocks[2][0]
# M10 = d.as_backend_type(d.assemble_mixed(J10))
# J110, J111 = Jblocks[3]
# M110 = d.as_backend_type(d.assemble_mixed(J110))#; M111 =
# M110 += d.as_backend_type(d.assemble_mixed(J111))
# #M110.axpy(1, M111, False)

# # jnest = d.PETScNestMatrix([M000, M01, M10, M110])
# # pjnest = jnest.mat()
# # pjnest.getNestSubMatrix(0,0).size

# jnest = PETSc.Mat().createNest([[M000.mat(), M01.mat()], [M10.mat(), M110.mat()]])
# jnest = PETSc.Mat().createNest([[M00.mat(), M01.mat()], [M10.mat(), M11.mat()]])
# jnest.getNestSubMatrix(0,0).size

# dim = 2
# #Jdpetsc = [[None]*dim]*dim
# Jdpetsc = []
# Jpetsc = [[None]*dim]*dim
# for i in range(dim):
#     for j in range(dim):
#         ij = i*dim + j
#         #Jdpetsc[i][j] = d.PETScMatrix()
#         Jsum = d.as_backend_type(d.assemble_mixed(Jblocks[ij][0]))#, tensor=Jdpetsc[i][j])
#         for k in range(1,len(Jblocks[ij])):
#             print(f"k={k}")
#             Jsum += d.as_backend_type(d.assemble_mixed(Jblocks[ij][k]))#, tensor=Jdpetsc[i][j])

#         Jpetsc[i][j] = Jsum#Jdpetsc[i][j].mat()
#         Jdpetsc.append(Jsum)#Jdpetsc[i][j].mat()
#         #print(f"init i={i}, j={j}: size={Jpetsc[i][j].size}")

# Jdpetsc_nest =  d.PETScNestMatrix([Jdpetsc[0], Jdpetsc[1], Jdpetsc[2], Jdpetsc[3]])
# Jpetsc_nest =  d.PETScNestMatrix([Jpetsc[0][0], Jpetsc[0][1], Jpetsc[1][0], Jpetsc[1][1]])
# #Jpetsc_nest =  PETSc.Mat().createNest(Jpetsc)

# dim=2
# for i in range(dim):
#     for j in range(dim):
#         print(f"{i}, {j} init")
#         print(Jpetsc_nest.getNestSubMatrix(i,j).size)


# dpM00 = d.PETScMatrix()
# dpM01 = d.PETScMatrix()
# M00 = d.assemble_mixed(J00, tensor=dpM00)
# M01 = d.assemble_mixed(J01, tensor=dpM01)
# M00.array().sum() # .04
# sum(M00.array()[:,0]) + sum(M00.array()[:,17]) # 0.0001
# sum(M01.array()[:,0]) + sum(M01.array()[:,17]) # 0.5723

# # add matrices
# # M00.axpy(1, M01, False)
# # M00.array().sum() # 8.04

# M00_copy = M00.copy()
# M00_copy.zero()
# J0 = M00_copy.mat()
# J0.axpy(1, M00.mat())
# J0.axpy(1, M01.mat())
# J0.getColumnVector(0).array.sum() + J0.getColumnVector(17).array.sum()
# sum(M00.array()[:,0]) + sum(M00.array()[:,17]) + sum(M01.array()[:,0]) + sum(M01.array()[:,17])

if False:
    # ====================================
    # Mixed problem
    # ====================================
    Fsum = sum([f.lhs for f in model.forms])  # Sum of all forms
    u = model.u["u"]

    new_Jlist = list()
    for Jij_domains in model.Jlist:
        Jij = [d.assemble_mixed(Jij_domain) for Jij_domain in Jij_domains]
        Jsum = Jij[0]
        if len(Jij) > 1:
            for domain_idx in range(1, len(Jij)):
                Jsum += Jij[domain_idx]

        new_Jlist.append(Jsum)

    b = d.PETScVector()  # same as b = PETSc.Vec()
    J_mat = d.PETScNestMatrix()
    myproblem = mySNESProblem(Fsum, u, model.Jlist)
    snes = PETSc.SNES().create(d.MPI.comm_world)
    snes.setFunction(myproblem.F, b.vec())
    snes.setJacobian(myproblem.J, J_mat.mat())

    sol = PETSc.Vec().createNest(
        [u.sub(0).vector().vec().copy(), u.sub(1).vector().vec().copy()],
    )
    # snes.solve(None, sol)

    Jnest = d.PETScNestMatrix(new_Jlist)
    unest = d.Vector()
    Jnest.init_vectors(unest, [u.sub(0).vector(), u.sub(1).vector()])

    # NestMatrix
    # dP -> P ->
    # Jnest00 = new_Jlist[0]
    Jnest = d.PETScNestMatrix(new_Jlist)
    pnestmat = Jnest.copy().mat()

    def assemble_submatrix(Jinput):
        # Matrix using assemble_mixed()
        # dolfin.Matrix -> petsc.Matrix -> d.PETScMatrix -> assemble again
        # Jmat = d.assemble_mixed(Jinput)
        # print(Jmat.array().sum()) # for Jinput=model.Jlist[0][0], this is 20

        #
        Jmat = d.PETScMatrix()
        # pJmat = d.as_backend_type(Jmat).mat()
        # dpJmat = d.PETScMatrix(pJmat)
        # model.dolfin_set_function_values(model.sc['A'], 'u', 33)
        # Jmat.zero()
        d.assemble_mixed(Jinput, tensor=Jmat)  # this updates Jmat

    assemble_submatrix(model.Jlist[0][0])

    # Starting from a petsc.NestMatrix, wrap with dolfin, then assemble submatrices
    petsc_mat = Jnest.mat()
    # Jmat = d.assemble_mixed(Jinput)
    # print(Jmat.array().sum()) # for Jinput=model.Jlist[0][0], this is 20

    # Vector
    # dolfin assemble -> petscvec -> dolfin wrapped petsvec -> assemble using dolfin wrapped petscvec as tensor
    # dolfin.Vector -> petsc.Vec -> d.PETScVec -> assemble again
    # Fvec = d.assemble(F)
    # sum(Fvec) # == -0.22990547835122357
    # pFvec = d.as_backend_type(Fvec).vec()
    # dpFvec = d.PETScVector(pFvec)
    # d.assemble(3*F, tensor=dpFvec) # this updates Fvec

    # def F(self, snes, x, F):
    #     # convert petsc type vectors into dolfin wrappers
    #     # petsc4py.PETSc.Vec -> d.PETScVector
    #     # e.g.
    #     # vec  = d.PETScVector(d.MPI.comm_world, 5)
    #     # pvec = vec.copy().vec()
    #     # d.PETScVector(pvec) works
    #     x = d.PETScVector(x)
    #     F  = d.PETScVector(F)

    #     # order matters
    #     # update "x" with values of u
    #     x.vec().copy(self.u.vector().vec())
    #     self.u.vector().apply("")
    #     d.assemble(self.L, tensor=F)
    #     # F = d.as_backend_type(d.assemble(self.L))

    #     for bc in self.bcs:
    #         bc.apply(F, x)

    # from dolfin import *
    # def problem():
    #     mesh = UnitCubeMesh(24, 16, 16)
    #     V = VectorFunctionSpace(mesh, "Lagrange", 1)

    #     # Mark boundary subdomians
    #     left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
    #     right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)

    #     # Define Dirichlet boundary (x = 0 or x = 1)
    #     c = Expression(("0.0", "0.0", "0.0"), degree=2)
    #     r = Expression(("scale*0.0",
    #                     "scale*(y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta) - x[1])",
    #                     "scale*(z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta) - x[2])"),
    #                     scale = 0.5, y0 = 0.5, z0 = 0.5, theta = pi/3, degree=2)

    #     bcl = DirichletBC(V, c, left)
    #     bcr = DirichletBC(V, r, right)
    #     bcs = [bcl, bcr]

    #     # Define functions
    #     du = TrialFunction(V)            # Incremental displacement
    #     v  = TestFunction(V)             # Test function
    #     u  = Function(V)                 # Displacement from previous iteration
    #     B  = Constant((0.0, -0.5, 0.0))  # Body force per unit volume
    #     T  = Constant((0.1,  0.0, 0.0))  # Traction force on the boundary

    #     # Kinematics
    #     d = u.geometric_dimension()
    #     I = Identity(d)             # Identity tensor
    #     F = I + grad(u)             # Deformation gradient
    #     C = F.T*F                   # Right Cauchy-Green tensor

    #     # Invariants of deformation tensors
    #     Ic = tr(C)
    #     J  = det(F)

    #     # Elasticity parameters
    #     E, nu = 10.0, 0.3
    #     mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

    #     # Stored strain energy density (compressible neo-Hookean model)
    #     psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

    #     # Total potential energy
    #     Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds

    #     # Compute first variation of Pi (directional derivative about u in the direction of v)
    #     F = derivative(Pi, u, v)

    #     return F, u, bcs

    # class SNESProblem():
    #     def __init__(self, F, u, bcs):
    #         V = u.function_space()
    #         du = d.TrialFunction(V)
    #         self.L = F
    #         self.a = d.derivative(F, u, du)
    #         self.bcs = bcs

    #         # because this is from a mixedfunctionspace
    #         self.u = u
    #     def F(self, snes, x, F):
    #         print(f"\n\n")
    #         print(f"type(F) = {type(F)}")
    #         x = d.PETScVector(x)
    #         F  = d.PETScVector(F)
    #         print(f"type(F) = {type(F)}")

    #         print(f"\n pre: ")
    #         print(f"x.max = {x.max()}")
    #         print(f"u.max = {self.u.vector().vec().max()}")
    #         x.vec().copy(self.u.vector().vec())
    #         print(f"\n mid: ")
    #         print(f"x.max = {x.max()}")
    #         print(f"u.max = {self.u.vector().vec().max()}")
    #         self.u.vector().apply("")
    #         print(f"\n post: ")
    #         print(f"x.max = {x.max()}")
    #         print(f"u.max = {self.u.vector().vec().max()}")

    #         # order matters
    #         d.assemble(self.L, tensor=F)
    #         # F = d.as_backend_type(d.assemble(self.L))

    #         for bc in self.bcs:
    #             bc.apply(F, x)
    #             # bc.apply(F, self.u.vector())
    #     # def F(self, snes, x, F):
    #     #     x = d.PETScVector(x)
    #     #     F  = d.PETScVector(F)
    #     #     x.vec().copy(self.u.vector().vec())
    #     #     self.u.vector().apply("")
    #     #     d.assemble(self.L, tensor=F)
    #     #     for bc in self.bcs:
    #     #         bc.apply(F, x)
    #     #         bc.apply(F, self.u.vector())

    #     def J(self, snes, x, J, P):
    #         J = d.PETScMatrix(J)
    #         x.copy(self.u.vector().vec())
    #         self.u.vector().apply("")
    #         d.assemble(self.a, tensor=J)
    #         for bc in self.bcs:
    #             bc.apply(J)

    # # # simple non-linear
    # mesh = d.UnitSquareMesh(30,30)
    # V = d.FunctionSpace(mesh, "CG", 1)
    # u = d.Function(V)
    # v = d.TestFunction(V)
    # f = d.Expression("x[0]*sin(x[1])", degree=1)
    # F = d.inner((1 + u**2)*d.grad(u), d.grad(v))*d.dx - f*v*d.dx
    # # Sub domain for Dirichlet boundary condition
    # class DirichletBoundary(d.SubDomain):
    #     def inside(self, x, on_boundary):
    #         return abs(x[0] - 1.0) < d.DOLFIN_EPS and on_boundary

    # #Define boundary condition
    # g = d.Constant(1.0)
    # bc = d.DirichletBC(V, g, DirichletBoundary())

    # #F, u, bcs = problem()
    # snesproblem = SNESProblem(F, u, [bc])
    # b = d.PETScVector()  # same as b = PETSc.Vec()
    # J_mat = d.PETScMatrix()

    # snes = PETSc.SNES().create(d.MPI.comm_world)
    # snes.setFunction(snesproblem.F, b.vec())
    # snes.setJacobian(snesproblem.J, J_mat.mat())
    # snes.solve(None, snesproblem.u.vector().vec())
    # # u.vector().vec().max() -> (27, 1.041746041109839)
