import stubs
import stubs.common as common
import dolfin as d
import ufl
import numpy as np
import pint
unit = stubs.unit # unit registry
#====================
# aliases - unit registry
# ===================
from stubs.model_assembly import Parameter, Species, Compartment, Reaction
uM       = unit.uM
meter    = unit.m
um       = unit.um
molecule = unit.molecule
sec      = unit.s

def make_model(mesh_name):

    #====================
    # define the model
    # ===================
    pc, sc, cc, rc = common.empty_sbmodel()
    # parameters
    pc.add([    
        Parameter('kf'      , 5.0, meter/sec, 'forward rate'),
        # # volume-to-volume [m/s]
        # Parameter('testing', 5, 1/sec),
        Parameter.from_expression('gating_f' , '5.0+t', um/sec, use_preintegration=False),
        Parameter('gating_r'      , 1.0, um/sec),#um/sec, 'reverse rate'),
        # volume mass-action 2 to 1
        Parameter('kf', 13.0, 1/(uM*sec), 'volume mass-action forward A+A2 -> A3'),
        Parameter('kr', 2.0, 1/(sec), 'volume mass-action reverse A3 -> A+A2'),
        # # volume to surface / surface to volume
        Parameter('kf_AX_X2', 3.0, 1/(uM*sec), 'A+X -> X2'),
        Parameter('kr_AX_X2', 3.0, 1/(sec), 'X2 -> A+X'),
        # # volume-surface to volume 
        Parameter('kf_AY_B', 3.0, 1/(uM*sec), 'A+Y -> B'),
        # # volume-volume to surface 
        Parameter('kf_AB_Y', 3.0, um/(uM*sec), 'A+B -> Y'),
        # # volume/surface degradation [1/s]
        Parameter('kdeg_B', 2.0, 1/sec, 'degradation rate'),
        Parameter('kdeg_X', 2.0, 1/sec, 'degradation rate'),
        Parameter('kdeg_Y', 2.0, 1/sec, 'degradation rate'),
    ])

    # species
    sc.add([
        Species('A'   , '10+z' , uM            , 100, um**2/sec, 'cytosol'),
        # Species('A'    , 'z'  , uM            , 100, um**2/sec, 'cytosol'),
        #Species('A2'   , '3+z', uM            , 100, um**2/sec, 'cytosol'),
        Species('A2'   , '5+z', uM            , 100, um**2/sec, 'cytosol'),
        Species('A3'   , '7+z'    , uM            , 100, um**2/sec, 'cytosol'),
        Species('B' , 3    , uM            , 100, um**2/sec, 'er_vol'),
        Species('X' , '100+z'  , molecule/um**2, 10 , um**2/sec, 'pm'),
        Species('X2' , 40  , molecule/um**2, 10 , um**2/sec, 'pm'),
        # Species('Y', 60  , molecule/um**2, 10 , um**2/sec, 'er_mem'),
    ])

    # compartments
    cc.add([
        Compartment('cytosol', 3, um, 11),
        Compartment('er_vol' , 3, um, 12),
        #Compartment('test_vol' , 3, um, [3,5]),
        Compartment('pm'     , 2, um, 2),
        Compartment('er_mem' , 2, um, 4),
    ])

    # flux topologies are commented
    rc.add([
        Reaction('A <-> B'      , ['A']     , ['B'] , {'on': 'gating_f', 'off': 'gating_r'} , explicit_restriction_to_domain='er_mem'), # [volume_to_volume] 
        Reaction('A + A2 <-> A3', ['A','A2'], ['A3'], {'on': 'kf',       'off': 'kr'}                                                ), # [volume] volume mass action (2 to 1)
        # Reaction('B -> 0'       , ['B']     , []    , {'on': 'kdeg_B'}                      , reaction_type='mass_action_forward'    ), # [volume] degradation
        Reaction('A + X <-> X2' , ['A','X'] , ['X2'], {'on': 'kf_AX_X2', 'off': 'kr_AX_X2'}                                          ), # [volume_to_surface] [surface_to_volume]
        # Reaction('X -> 0'       , ['X']     , []    , {'on': 'kdeg_X'}                      , reaction_type='mass_action_forward'    ), # [surface] degradation
        # Reaction('Y -> 0'       , ['Y']     , []    , {'on': 'kdeg_Y'}                      , reaction_type='mass_action_forward'    ), # [surface] degradation
        # Reaction('A + Y <-> B'  , ['A','Y'] , ['B'] , {'on': 'kf_AY_B'}                     , reaction_type='mass_action_forward'    ), # [volume-surface_to_volume]
        # Reaction('A + B <-> Y'  , ['A','B'] , ['Y'] , {'on': 'kf_AB_Y'}                     , reaction_type='mass_action_forward'    ), # [volume-volume_to_surface]
    ])

    # config
    stubs_config = stubs.config.Config()
    stubs_config.flags['allow_unused_components'] = True

    solver_parameters = {'final_t':0.1, 'initial_dt':0.01, 'dt_increase_factor':1.1}
    stubs_config.solver.update(solver_parameters)
    # stubs_config.loglevel['dolfin'] = 'CRITICAL'
    # Define solvers
    # mps           = stubs.solvers.MultiphysicsSolver()
    # nls           = stubs.solvers.NonlinearNewtonSolver()
    # ls            = stubs.solvers.DolfinKrylovSolver()
    # solver_system = stubs.solvers.SolverSystem(final_t=0.1, initial_dt=0.01, multiphysics_solver=mps, nonlinear_solver=nls, linear_solver=ls)
    # mesh
    path = stubs.common.data_path()
    
    if mesh_name[-2:] == 'h5':
        stubs_mesh = stubs.mesh.ParentMesh(str(path / mesh_name), 'hdf5')
    else:
        stubs_mesh = stubs.mesh.ParentMesh(str(path / mesh_name), 'xml')

    return stubs.model.Model(pc, sc, cc, rc, stubs_config, stubs_mesh)
