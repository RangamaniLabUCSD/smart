# import 
from dolfin import * # fenics/dolfin api
import mpi4py.MPI as pyMPI

import stubs
from stubs import unit as ureg


# ====================================================
# ====================================================
comm = MPI.comm_world
rank = comm.rank
root = 0
nprocs = comm.size

# Load in model and settings
config = stubs.config_refactor.ConfigRefactor()
PD = stubs.common.json_to_ObjectContainer('toy_model/parameters.json', 'parameters')
SD = stubs.common.json_to_ObjectContainer('toy_model/species.json', 'species')
CD = stubs.common.json_to_ObjectContainer('toy_model/compartments.json', 'compartments')
RD = stubs.common.json_to_ObjectContainer('toy_model/reactions.json', 'reactions')

# example call
mps = stubs.solvers.MultiphysicsSolver('iterative')
nls = stubs.solvers.NonlinearNewtonSolver()
ls = stubs.solvers.DolfinKrylovSolver()
solver_system = stubs.solvers.SolverSystem(multiphysics_solver=mps, nonlinear_solver=nls, linear_solver=ls)

model = stubs.model_refactor.ModelRefactor(PD, SD, CD, RD, config, solver_system)

#settings = stubs.config.Config('main.config')
#model = settings.generate_model()


#model.solve()

