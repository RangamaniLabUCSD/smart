# import 
from dolfin import * # fenics/dolfin api
import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc
Print = PETSc.Sys.Print
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
settings = stubs.config.Config('main.config')
model = settings.generate_model()

# to deal with possible floating point error in mesh coordinates
model.set_allow_extrapolation()
# Turn fluxes into fenics/dolfin expressions
model.assemble_reactive_fluxes()
model.assemble_diffusive_fluxes()
model.establish_mappings()
# Sort forms by type (diffusive, time derivative, etc.)
model.sort_forms()



## solve
model.init_solver_and_plots()

solver_idx=0
model.stopwatch("Total simulation")
while True:
    solver_idx +=1
    model.iterative_solver(boundary_method='RK45')
    model.compute_statistics()
    if model.idx % 5 == 0:
        model.plot_solution()
        model.plot_solver_status()
    if model.t >= model.config.solver['T'] or solver_idx>5000:
        model.plot_solution()
        model.plot_solver_status()
        break

model.stopwatch("Total simulation", stop=True)
Print("Solver finished with %d total time steps." % int(solver_idx))


