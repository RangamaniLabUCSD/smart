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


model.solve()

