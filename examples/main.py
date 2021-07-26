# import 
import dolfin as d # dolfin/fenics api
#import mpi4py.MPI as pyMPI
import stubs
import os
from stubs import unit as ureg

# ====================================================
# ====================================================
comm = d.MPI.comm_world
rank = comm.rank
root = 0
nprocs = comm.size
cwd = os.getcwd()

# Load in model and settings
config = stubs.config.Config()
PD, SD, CD, RD = stubs.common.read_smodel('toy_model/cell2013_3d.smodel')

# Define solvers
mps = stubs.solvers.MultiphysicsSolver('iterative', eps_Fabs=1e-8)
nls = stubs.solvers.NonlinearNewtonSolver(relative_tolerance=1e-6, absolute_tolerance=1e-8,
                                          dt_increase_factor=1.05, dt_decrease_factor=0.7)
ls = stubs.solvers.DolfinKrylovSolver(method = 'bicgstab', preconditioner='hypre_amg')
solver_system = stubs.solvers.SolverSystem(final_t = 0.4, initial_dt = 0.01, adjust_dt = [(0.2, 0.02)],
                                           multiphysics_solver=mps, nonlinear_solver=nls, linear_solver=ls)

cyto_mesh = stubs.mesh.Mesh(mesh_filename=cwd+'/cell2013_3d/cube_10.xml', name='cyto')

model = stubs.model.Model(PD, SD, CD, RD, config, solver_system, cyto_mesh)
model.initialize()

# solve system
model.solve(store_solutions=False)
#model.solve_zero_d([0,1])
