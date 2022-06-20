# debugging 06/04/22


import petsc4py.PETSc as p
import dolfin as d
import matplotlib.pyplot as plt
comm = d.MPI.comm_world

P = d.PETScMatrix()
mesh = d.UnitSquareMesh(2,2)
V = d. FunctionSpace(mesh, 'CG', 1)
u = d.TrialFunction(V)
u_ = d.interpolate(d.Expression('10*x[0]*x[1]', degree=1), V)
v = d.TestFunction(V)
F = d.inner(d.grad(u), d.grad(v))*d.dx 
L = d.inner(u_, v)*d.dx
B = d.assemble(L)
M = d.assemble(F)
# M.local_range(0)
Mback = d.as_backend_type(M).mat()
Bback = d.as_backend_type(B).vec()

lnrow = gnrow = lncol = gncol = 121

Q = p.Mat().create(comm=comm)
# ((local_nrows, global_nrows), (local_ncols, global_ncols))
Q.setSizes(((lnrow, gnrow), (lncol, gncol)))
Q.setType("aij")
Q.setUp()

for i,mat in enumerate([Mback, Q]): 
    print(f"\n\n===========================\nMatrix:  {['Mback', 'Q'][i]} ")
    print(mat.getSizes())
    print(mat.getType())
    print(mat.getOwnershipRange())
    print(mat.getLocalSize())
    print(mat.getOwnershipIS()[0].array)
    # print(mat.getLGMap()[0].indices)

# # Q.setUp()
# if lgmap is not None:
#     M.setLGMap(lgmap, lgmap)
# if assemble:
#     M.assemble()


# i=0;j=0;k=0
# dolfin_map = p.LGMap().create(model.problem.Jforms_all[0][0].function_space(0).dofmap().dofs(), comm=model.problem.mpi_comm_world)
# dolfinmap.indices
# N = model.problem.local_block_sizes[0] # same as global for ncpu=1
# M0=model.problem.init_zero_petsc_matrix(N,N,N,N)

# M = d.assemble_mixed(model.problem.Jforms_all[0][0])
# M_ = d.as_backend_type(M).mat() # M_.getType() == 'seqaij'

# lg0 = M0.getLGMap()[0]
# lg0.block_indices


# class foo:
#     def __init__(self):
#         self.a=1

#     def bar(self, a):
#         a += 3
#         print('hi')