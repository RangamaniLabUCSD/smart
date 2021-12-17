import stubs
import pytest
import pint
import math

from stubs.common import create_sbmodel

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

    sbmodel = stubs.common.create_sbmodel(p,s,c,r)
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
def test_stubs_model_init(stubs_model):
    "Test the different parts of model initialization"
    # initialize
    model = stubs_model
    assert type(model)==stubs.model.Model

    model._init_1()
    model._init_2()
    model._init_3()

    # aliases
    cm = model.child_meshes['pm']
    m  = model.parent_mesh

    for idx in [1, 7, 14]:
        # test child cell -> parent entity mapping
        a  = cm.map_cell_to_parent_vertex[idx,:]
        pidx = cm.map_cell_to_parent_entity[idx]
        b    = m.facets[pidx]
        assert all(a==b)
        assert all(cm.cell_coordinates[idx] == m.facet_coordinates[pidx])

        # test child facet -> parent entity mapping
        pidx=cm.map_facet_to_parent_entity[idx]
        assert all(cm.map_facet_to_parent_vertex[idx,:] == pm.facets[pidx,:])
    
    # check volumes and surfaces
    assert math.isclose(cm.get_nvolume('dx', 12), 8.0)
    assert math.isclose(cm.get_nvolume('ds', 2), 20.0)
    assert math.isclose(cm.get_nvolume('ds', 4), 4.0)
