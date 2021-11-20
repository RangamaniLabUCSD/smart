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

import os
cwd=os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append("../../")
import stubs
unit = stubs.unit # unit registry

for meshfile in ['/Users/rabona/Documents/stubs/examples/cell2013_3d/cube_10.xml', './unit_cube.xml']:
    for kf, kr in [(5.0,0.1),(10.0,1.0),(2.0,0.25)]:
        for unitA in [unit.molecule/unit.um**3, unit.uM]:
            for init_A, init_B, init_C in [(10,1000,0),(12,20,1),(20,10,5)]:
                for diff_A, diff_B, diff_C in [(10,0.1,50),(5,5,10),(1,10,10)]:
                    # initialize 
                    p = stubs.model_building.ParameterDF()
                    s = stubs.model_building.SpeciesDF()
                    c = stubs.model_building.CompartmentDF()
                    r = stubs.model_building.ReactionDF()

                    ### define parameters
                    # name, value, unit, notes
                    p.append('kf', kf, 1/(unit.uM*unit.s), 'forward rate')
                    p.append('kr', kr, 1/unit.s, 'reverse rate')

                    ### define species
                    # name, plot group, concentration units, initial condition, diffusion
                    # coefficient, diffusion coefficient units, compartment
                    s.append('A', 'cytosolic', unitA, init_A, diff_A, unit.um**2/unit.s, 'cyto')
                    s.append('X', 'membrane bound', unit.molecule/unit.um**2, init_C, diff_B, unit.um**2/unit.s, 'pm')
                    s.append('B', 'membrane bound', unit.molecule/unit.um**2, init_B, diff_C, unit.um**2/unit.s, 'pm')

                    ### define compartments
                    # name, geometric dimensionality, length scale units, marker value
                    c.append('cyto', 3, unit.um, 1)
                    c.append('pm', 2, unit.um, 2)

                    ### define reactions
                    # name, notes, left hand side of reaction, right hand side of reaction, kinetic
                    # parameters
                    r.append('A+X <-> B', 'cell 2013', ['A','X'], ['B'], {"on": "kf", "off": "kr"})


                    # write out to file
                    stubs.common.write_sbmodel(cwd + '/cell2013_3d.sbmodel', p, s, c, r)
                    # Load in model and settings
                    config = stubs.config.Config()
                    #pc = stubs.common.json_to_ObjectContainer('toy_model_2d/parameters.json', 'parameters')
                    #sc = stubs.common.json_to_ObjectContainer('toy_model_2d/species.json', 'species')
                    #cc = stubs.common.json_to_ObjectContainer('toy_model_2d/compartments.json', 'compartments')
                    #rc = stubs.common.json_to_ObjectContainer('toy_model_2d/reactions.json', 'reactions')
                    sbmodel = stubs.common.read_sbmodel('cell2013_3d/cell2013_3d.sbmodel')

                    # Define solvers
                    mps = stubs.solvers.MultiphysicsSolver('iterative', eps_Fabs=1e-8)
                    nls = stubs.solvers.NonlinearNewtonSolver(relative_tolerance=1e-6, absolute_tolerance=1e-8,
                                                            dt_increase_factor=1.05, dt_decrease_factor=0.7)
                    ls = stubs.solvers.DolfinKrylovSolver(method = 'bicgstab', preconditioner='hypre_amg')
                    solver_system = stubs.solvers.SolverSystem(final_t = 0.4, initial_dt = 0.01, adjust_dt = [(0.2, 0.02)],
                                                            multiphysics_solver=mps, nonlinear_solver=nls, linear_solver=ls)
                    cyto_mesh = stubs.mesh.Mesh(mesh_filename=meshfile, name='cyto')

                    model = stubs.model.Model(sbmodel, config, solver_system, cyto_mesh)
                    model.initialize()
                    model.solve_2(check_mass=True, species_to_check={'A':1,'X':1,'B':2}, x_compartment='cyto')


# solve system

