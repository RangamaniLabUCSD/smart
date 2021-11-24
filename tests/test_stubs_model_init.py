# import stubs
# import pytest

# # Fixtures
# @pytest.fixture
# def stubs_model():
#     unit = stubs.unit # unit registry

#     # initialize 
#     p, s, c, r = stubs.model_building.empty_sbmodel()

#     ## define parameters
#     # name, value, unit, notes
#     p.append('kdeg', 5.0, 1/(unit.s), 'degradation rate')

#     ## define species
#     # name, plot group, concentration units, initial condition, diffusion
#     # coefficient, diffusion coefficient units, compartment
#     s.append('B', 'cytosolic', unit.uM, 10, 1, unit.um**2/unit.s, 'cyto')

#     ## define compartments
#     # name, geometric dimensionality, length scale units, marker value
#     c.append('cyto', 3, unit.um, 1)

#     ## define reactions
#     # name, notes, left hand side of reaction, right hand side of reaction, kinetic
#     # parameters
#     r.append('B linear degredation', 'example reaction', ['B'], [], {"on": "kdeg"}, reaction_type='mass_action_forward')

#     sbmodel = stubs.common.read_sbmodel('pytest.sbmodel')
#     # Define solvers
#     mps             = stubs.solvers.MultiphysicsSolver()
#     nls             = stubs.solvers.NonlinearNewtonSolver()
#     ls              = stubs.solvers.DolfinKrylovSolver()
#     solver_system   = stubs.solvers.SolverSystem(final_t=0.1, initial_dt=0.01, multiphysics_solver=mps, nonlinear_solver=nls, linear_solver=ls)

#     model = stubs.model.Model(sbmodel, stubs_config, solver_system, parent_mesh=stubs_mesh)

#     assert type(model)==stubs.model.Model
#     model = 
#     return stubs.mesh.ParentMesh(mesh_filename=mesh_filename)

# @pytest.fixture
# def stubs_config():
#     return stubs.config.Config()

# # Tests
# @pytest.mark.stubs_model_init
# def test_stubs_mesh_load_dolfin_mesh(stubs_mesh):
#     "Make sure that stubs is loading the dolfin mesh when we create a ParentMesh"
#     assert stubs_mesh.dolfin_mesh.num_vertices() > 1
#     assert stubs_mesh.dolfin_mesh.num_cells() > 1