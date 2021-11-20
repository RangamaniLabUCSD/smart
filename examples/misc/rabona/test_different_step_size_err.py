
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

errors=[]
for dt in [0.012, 0.01, 0.008, 0.006, 0.004, 0.002]:
    pc, sc, cc, rc = stubs.common.read_sbmodel('model.sbmodel')

    # Define solvers
    mps = stubs.solvers.MultiphysicsSolver('iterative')
    nls = stubs.solvers.NonlinearNewtonSolver(relative_tolerance=1e-9, absolute_tolerance=1e-9,
                                            dt_increase_factor=1.0, dt_decrease_factor=1.0)
    ls = stubs.solvers.DolfinKrylovSolver(method = 'gmres', preconditioner='ilu')


    solver_system = stubs.solvers.SolverSystem(final_t = 0.4, initial_dt = dt, adjust_dt = [],
                                            multiphysics_solver=mps, nonlinear_solver=nls, linear_solver=ls)
    cyto_mesh = stubs.mesh.Mesh(mesh_filename='/Users/rabona/Documents/stubs/examples/unit_cube.xml', name='cyto')

    model = stubs.model_refactor.ModelRefactor(pc, sc, cc, rc, config, solver_system, cyto_mesh)
    model.initialize()

    # solve system
    model.solve()
    errors.append(np.max(model.u['cyto']['u'].vector().get_local()-(lambda t:10*np.exp(-5*t))(model.t))/(lambda t:10*np.exp(-5*t))(model.t))

