import pytest
import math

import stubs
from stubs.model_assembly import (
    Parameter,
    Species,
    Compartment,
    Reaction,
    empty_sbmodel,
)

# Fixtures


@pytest.fixture(name="model")
def stubs_model(stubs_mesh):
    # initialize
    pc, sc, cc, rc = empty_sbmodel()
    # units
    unit = stubs.unit  # unit registry
    uM = unit.uM
    meter = unit.m
    um = unit.um
    molecule = unit.molecule
    sec = unit.s

    # parameters
    pc.add(
        [
            Parameter("kf", 5.0, meter / sec, "forward rate"),
            # Parameter.from_expression('kf_t' , '5.0+t', meter/sec),
            Parameter("kr", 1.0, um / sec, "reverse rate"),
            Parameter("kdeg_B", 2.0, 1 / sec, "degradation rate"),
            Parameter("kdeg_Xpm", 2.0, 1 / sec, "degradation rate"),
            Parameter("kdeg_Xerm", 2.0, 1 / sec, "degradation rate"),
        ]
    )

    # species
    sc.add(
        [
            Species("A", 10, uM, 100, um**2 / sec, "cytosol"),
            # Species('A'   , '5*(x+1)+2' , uM            , 100, um**2/sec, 'cytosol'),
            Species("B", 10, uM, 100, um**2 / sec, "cytosol"),
            Species("A_er", 3, uM, 100, um**2 / sec, "er_vol"),
            Species("X_pm", 100, molecule / um**2, 10, um**2 / sec, "pm"),
            Species("X_erm", 100, molecule / um**2, 10, um**2 / sec, "er_mem"),
        ]
    )

    # compartments
    cc.add(
        [
            Compartment("cytosol", 3, um, 11),
            Compartment("er_vol", 3, um, 12),
            Compartment("pm", 2, um, 2),
            Compartment("er_mem", 2, um, 4),
        ]
    )

    rc.add(
        [
            Reaction("A <-> A_er", ["A"], ["A_er"], {"on": "kf", "off": "kr"}),
            Reaction(
                "B -> 0",
                ["B"],
                [],
                {"on": "kdeg_B"},
                reaction_type="mass_action_forward",
            ),
            Reaction(
                "X_pm -> 0",
                ["X_pm"],
                [],
                {"on": "kdeg_Xpm"},
                reaction_type="mass_action_forward",
            ),
            Reaction(
                "X_erm -> 0",
                ["X_erm"],
                [],
                {"on": "kdeg_Xerm"},
                reaction_type="mass_action_forward",
            ),
        ]
    )

    # config (FFC logger breaks older versions of pytest)
    stubs_config = stubs.config.Config()
    stubs_config.loglevel.FFC = "ERROR"
    stubs_config.loglevel.UFL = "ERROR"
    stubs_config.loglevel.dolfin = "ERROR"

    stubs_config.solver.update(
        {
            "final_t": 1,
            "initial_dt": 0.01,
            "time_precision": 6,
            "use_snes": True,
            "print_assembly": False,
        }
    )

    # Define solvers
    # FIXME: None of these solvers are defined
    # mps = stubs.solvers.MultiphysicsSolver()
    # nls = stubs.solvers.NonlinearNewtonSolver()
    # ls = stubs.solvers.DolfinKrylovSolver()
    # solver_system = stubs.solvers.SolverSystem(final_t=0.1, initial_dt=0.01)

    model = stubs.model.Model(pc, sc, cc, rc, stubs_config, stubs_mesh)

    return model


# Tests
@pytest.mark.xfail
@pytest.mark.stubs_model_init
def test_stubs_model_init(model):
    "Test the different parts of model initialization"
    # initialize
    model.initialize(initialize_solver=False)
    model.initialize_discrete_variational_problem_and_solver()
    # model._init_1()
    # model._init_2()
    # model._init_3()
    # model._init_4()

    # aliases
    pm_mesh = model.child_meshes["pm"]
    parent_mesh = model.parent_mesh

    test_indices = [1, 7, 12]  # some random indices
    # test child cell -> parent entity mapping
    for idx in test_indices:  # few different indices
        a = pm_mesh.map_cell_to_parent_vertex[idx, :]
        pidx = pm_mesh.map_cell_to_parent_entity[idx]
        b = parent_mesh.facets[pidx]
        assert (a == b).all()
        assert (pm_mesh.cell_coordinates[idx] == parent_mesh.facet_coordinates[pidx]).all()

    cyto_mesh = model.child_meshes["cytosol"]
    for idx in test_indices:
        # test child facet -> parent entity mapping
        pidx = cyto_mesh.map_facet_to_parent_entity[idx]
        assert all(cyto_mesh.map_facet_to_parent_vertex[idx, :] == parent_mesh.facets[pidx, :])

    # check volumes and surfaces
    assert math.isclose(parent_mesh.get_nvolume("dx"), 16.0)
    assert math.isclose(cyto_mesh.get_nvolume("dx"), 8.0)
    assert math.isclose(cyto_mesh.get_nvolume("ds", 2), 20.0)
    assert math.isclose(cyto_mesh.get_nvolume("ds", 4), 4.0)
    assert math.isclose(pm_mesh.get_nvolume("dx"), 40.0)
