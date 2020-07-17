# import 
import dolfin as d # dolfin/fenics api
#import mpi4py.MPI as pyMPI
import stubs
from stubs import unit as ureg

# ====================================================
# ====================================================
comm = d.MPI.comm_world
rank = comm.rank
root = 0
nprocs = comm.size

# Load in model and settings
config = stubs.config.Config()
#PD = stubs.common.json_to_ObjectContainer('toy_model_2d/parameters.json', 'parameters')
#SD = stubs.common.json_to_ObjectContainer('toy_model_2d/species.json', 'species')
#CD = stubs.common.json_to_ObjectContainer('toy_model_2d/compartments.json', 'compartments')
#RD = stubs.common.json_to_ObjectContainer('toy_model_2d/reactions.json', 'reactions')
PD, SD, CD, RD = stubs.common.read_smodel('model.smodel')

# Define solvers
mps = stubs.solvers.MultiphysicsSolver('iterative')
nls = stubs.solvers.NonlinearNewtonSolver(relative_tolerance=1e-6, absolute_tolerance=1e-8,
                                          dt_increase_factor=1.05, dt_decrease_factor=0.7)
ls = stubs.solvers.DolfinKrylovSolver(method = 'bicgstab', preconditioner='hypre_amg')
solver_system = stubs.solvers.SolverSystem(final_t = 0.4, initial_dt = 0.01, adjust_dt = [(0.2, 0.02)],
                                           multiphysics_solver=mps, nonlinear_solver=nls, linear_solver=ls)
cyto_mesh = stubs.mesh.Mesh(mesh_filename='/home/justin/gitrepos/stubs/examples/toy_model/cube_10.xml', name='cyto')

model = stubs.model.Model(PD, SD, CD, RD, config, solver_system, cyto_mesh)
model.initialize()

# solve system
model.solve()
