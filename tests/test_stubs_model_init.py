import stubs
import pytest
import pint

from stubs.common import init_sbmodel

# Fixtures
@pytest.fixture
def stubs_model(stubs_mesh, stubs_config):
    unit = stubs.unit # unit registry
    # initialize 
    p, s, c, r = stubs.model_building.empty_sbmodel()

    ## define parameters
    p.append('kf', 5.0, 1/(unit.s), 'foward rate')
    p.append('kr', 1.0, 1/(unit.s), 'foward rate')
    p.append('kdeg', 2.0, 1/(unit.s), 'degradation rate')

    ## define species
    # name, plot group, concentration units, initial condition, diffusion
    # coefficient, diffusion coefficient units, compartment
    s.append('A', 'cytosolic', unit.molecule/unit.um**3, 10.0, 10, unit.um**2/unit.s, 'cyto')
    s.append('X', 'membrane bound', unit.molecule/unit.um**2, 1000.0, 0.1, unit.um**2/unit.s, 'pm')
    s.append('B', 'membrane bound', unit.molecule/unit.um**2, 0.0, 50, unit.um**2/unit.s, 'pm')

    ## define compartments
    # name, geometric dimensionality, length scale units, marker value
    c.append('cyto', 3, unit.um, 1)
    c.append('pm', 2, unit.um, 2)

    ## define reactions
    # name, notes, left hand side of reaction, right hand side of reaction, kinetic parameters
    r.append('A+X <-> B', 'First reaction', ['A','X'], ['B'], {"on": "kf", "off": "kr"})
    r.append('B linear degredation', 'second reaction', ['B'], [], {"on": "kdeg"}, reaction_type='mass_action_forward')

    sbmodel = stubs.common.init_sbmodel(p,s,c,r)
    # Define solvers
    mps             = stubs.solvers.MultiphysicsSolver()
    nls             = stubs.solvers.NonlinearNewtonSolver()
    ls              = stubs.solvers.DolfinKrylovSolver()
    solver_system   = stubs.solvers.SolverSystem(final_t=0.1, initial_dt=0.01, multiphysics_solver=mps, nonlinear_solver=nls, linear_solver=ls)

    model = stubs.model.Model(sbmodel, stubs_config, solver_system, parent_mesh=stubs_mesh)

    return model

@pytest.fixture
def stubs_config():
    return stubs.config.Config()

# Tests
@pytest.mark.stubs_model_init
def test_stubs_model_init_part1_unit_assembly(stubs_model):
    "Assembling Pint units"
    # initialize
    model = stubs_model
    assert type(model)==stubs.model.Model

    # assemble the units
    model.pc.do_to_all('assemble_units', {'unit_name': 'unit'})
    model.pc.do_to_all('assemble_units', {'value_name':'value', 'unit_name':'unit', 'assembled_name': 'value_unit'})
    model.pc.do_to_all('assemble_time_dependent_parameters')
    model.sc.do_to_all('assemble_units', {'unit_name': 'concentration_units'})
    model.sc.do_to_all('assemble_units', {'unit_name': 'D_units'})
    model.cc.do_to_all('assemble_units', {'unit_name':'compartment_units'})
    model.rc.do_to_all('initialize_flux_equations_for_known_reactions', {"reaction_database": model.config.reaction_database})

    assert all([isinstance(param.value_unit, pint.Quantity) for param in model.pc.values])