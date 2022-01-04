"""
Solver classes. Linear/Nonlinear are wrappers around dolfin solvers.
MultiphysicsSolver uses a picard iteration scheme to solve coupled problems.
"""
import re
import os
from pandas import read_json
import dolfin as d
from stubs.common import nan_to_none
from stubs.common import round_to_n
from stubs import model_assembly
import random
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import petsc4py.PETSc as PETSc
Print = PETSc.Sys.Print

import mpi4py.MPI as pyMPI
comm = d.MPI.comm_world
rank = comm.rank
size = comm.size
root = 0

# Base Solver class
class Solver:
    def __init__(self, framework='dolfin'):
        self.framework = framework

    def check_validity(self):
        if self.framework not in ['dolfin']:
            raise ValueError(f"Framework {self.framework} is not supported.")

        print('Solver settings are valid')

# Base MultiphysicsSolver class
class MultiphysicsSolver(Solver):
    def __init__(self, method='iterative', eps_Fabs=1e-6, eps_udiff_abs=1e-8, eps_udiff_rel=1e-5,
                 min_multiphysics=2, max_multiphysics=6, dt_increase_factor=1.0, dt_decrease_factor=0.8):
        super().__init__()
        self.method             = method
        self.eps_Fabs           = eps_Fabs
        self.eps_udiff_abs      = eps_udiff_abs
        self.eps_udiff_rel      = eps_udiff_rel
        self.min_multiphysics   = min_multiphysics
        self.max_multiphysics   = max_multiphysics
        self.dt_increase_factor = dt_increase_factor
        self.dt_decrease_factor = dt_decrease_factor

    def check_validity(self):
        super().check_validity()
        if self.method not in ['iterative', 'monolithic']:
            raise ValueError(f"MultiphysicsSolver must be either 'iterative' or 'monolithic'.")

        print('All MultiphysicsSolver settings are valid!')

# Base NonlinearSolver class
class NonlinearSolver(Solver):
    def __init__(self, method='newton', min_nonlinear=2, max_nonlinear=10,
                 dt_increase_factor=1.0, dt_decrease_factor=0.8,):
        super().__init__()
        self.method=method
        self.min_nonlinear=min_nonlinear
        self.max_nonlinear=max_nonlinear
        self.dt_increase_factor=dt_increase_factor
        self.dt_decrease_factor=dt_decrease_factor
    def check_validity(self):
        # check settings of parent class
        super().check_validity()
        """
        Check that all current settings are valid
        """
        if type(self.min_nonlinear) != int or type(self.max_nonlinear) != int:
            raise TypeError("min_nonlinear and max_nonlinear must be integers.")

        if self.dt_increase_factor < 1.0: 
            raise ValueError("dt_increase_factor must be >= 1.0")

        print("All NonlinearSolver settings are valid!")

class NonlinearNewtonSolver(NonlinearSolver):
    """
    Settings for dolfin nonlinear Newton solver
    """
    def __init__(self, maximum_iterations=50, error_on_nonconvergence=False, relative_tolerance=1e-6, 
                 absolute_tolerance=1e-8, min_nonlinear=None, max_nonlinear=None, dt_increase_factor=None, dt_decrease_factor=None):
        super().__init__(method = 'newton')

        self.maximum_iterations         = maximum_iterations
        self.error_on_nonconvergence    = error_on_nonconvergence
        self.relative_tolerance         = relative_tolerance
        self.absolute_tolerance         = absolute_tolerance

        for prop_name, prop in {'min_nonlinear':min_nonlinear, 'max_nonlinear':max_nonlinear,
                                'dt_increase_factor':dt_increase_factor, 'dt_decrease_factor': dt_decrease_factor}.items():
            if prop is not None:
                setattr(self, prop_name, prop)

    def check_validity(self):
        super().check_validity()
        assert hasattr(self, 'framework')
        assert self.framework in ['dolfin']
        if type(self.maximum_iterations) != int:
            raise TypeError("maximum_iterations must be an int")
        if self.relative_tolerance <= 0: 
            raise ValueError("relative_tolerance must be greater than 0")
        if self.absolute_tolerance <= 0: 
            raise ValueError("absolute_tolerance must be greater than 0")

        print("All NonlinearNewtonSolver settings are valid!")

class NonlinearPicardSolver(NonlinearSolver):
    def __init__(self, picard_norm = 'Linf', maximum_iterations=50):
        super().__init__(method='picard')

        self.maximum_iterations         = maximum_iterations
        self.picard_norm = picard_norm

    def check_validity(self):
        super().check_validity()
        valid_norms = ['Linf', 'L2']
        if picard_norm not in valid_norms:
            raise ValueError(f"Invalid Picard norm ({picard_norm}) reasonable values are: {valid_norms}")

# Base LinearSolver class
class LinearSolver(Solver):
    def __init__(self, method='gmres', preconditioner='hypre_amg'):
        super().__init__()
        self.method         = method
        self.preconditioner = preconditioner
    def check_validity(self):
        super().check_validity()

class DolfinKrylovSolver(LinearSolver):
    def __init__(self, method = 'gmres', preconditioner = 'hypre_amg', 
                 maximum_iterations = 100000, error_on_nonconvergence = False,
                 nonzero_initial_guess = True, relative_tolerance = 1e-6,
                 absolute_tolerance = 1e-8):
        super().__init__(method = method, preconditioner = preconditioner)
        self.maximum_iterations         = maximum_iterations
        self.error_on_nonconvergence    = error_on_nonconvergence
        self.nonzero_initial_guess      = nonzero_initial_guess
        self.relative_tolerance         = relative_tolerance
        self.absolute_tolerance         = absolute_tolerance

    def check_validity(self):
        super().check_validity()
        if self.method not in d.krylov_solver_methods().keys():
            raise ValueError(f"Method {self.method} is not a supported dolfin krylov solver.")
        if self.preconditioner not in d.krylov_solver_preconditioners().keys():
            raise ValueError(f"Preconditioner {self.preconditioner} is not a supported dolfin krylov preconditioner.")


class SolverSystem:
    def __init__(self, final_t, initial_dt, multiphysics_solver=MultiphysicsSolver(), 
                 nonlinear_solver=NonlinearNewtonSolver(), linear_solver=DolfinKrylovSolver(), 
                 ignore_surface_diffusion=False, auto_preintegrate=True,
                 adjust_dt=None):

        self.final_t                    = final_t
        self.initial_dt                 = initial_dt
        self.multiphysics_solver        = multiphysics_solver
        self.nonlinear_solver           = nonlinear_solver
        self.linear_solver              = linear_solver

        # Sort adjust_dt if it was provided
        if adjust_dt is not None:
            self.check_adjust_dt_validity(adjust_dt)
            adjust_dt.sort(key = lambda tuple_: tuple_[0]) # sort by the first index (time to reset)
        self.adjust_dt                  = adjust_dt

        # General settings 
        self.ignore_surface_diffusion   = ignore_surface_diffusion # setting to True will treat surface variables as ODEs 
        self.auto_preintegrate          = auto_preintegrate

    def check_solver_system_validity(self):
        if self.final_t <= 0:
            raise ValueError("final_t must be > 0")
        if self.initial_dt <= 0:
            raise ValueError("initial_dt must be > 0")
        if self.initial_dt > self.final_t:
            raise ValueError("initial_dt must be < final_t")
            

        self.check_adjust_dt_validity(self.adjust_dt)
        if self.adjust_dt is not None and self.adjust_dt != sorted(self.adjust_dt, key = lambda tuple_: tuple_[0]):
            raise ValueError("adjust_dt not sorted...")

        for solver in [self.multiphysics_solver, self.nonlinear_solver, self.linear_solver]:
            if solver is not None:
                solver.check_validity()
            else:
                Print(f"Warning: SolverSystem does not include a {type(solver).__name__}.")

    def check_adjust_dt_validity(self, adjust_dt):
        # make sure adjust_dt is a list of 2-tuples with proper values
        # (time to reset, new dt)
        if adjust_dt is not None:
            if type(adjust_dt) is not list:
                raise ValueError("adjust_dt must be a list of 2-tuples")
            if len(adjust_dt) >= 1:
                for tuple_ in adjust_dt:
                    if type(tuple_) != tuple or len(tuple_) != 2:
                        raise ValueError("adjust_dt must be a list of 2-tuples")
                    if tuple_[0] < 0:
                        raise ValueError("First index of 2-tuples in adjust_dt (time to reset) must be >= 0")

    def make_dolfin_parameter_dict(self):
        """
        Helper function to package all necessary parameters into dictionaries which can be input as dolfin solver parameters
        """
        nonlinear_solver_keys = ['maximum_iterations', 'error_on_nonconvergence', 'absolute_tolerance', 'relative_tolerance']
        self.nonlinear_dolfin_solver_settings = {**{k:v for k,v in self.nonlinear_solver.__dict__.items() if k in nonlinear_solver_keys}, 
                                  **{'linear_solver': self.linear_solver.method,
                                     'preconditioner': self.linear_solver.preconditioner}}
        linear_solver_keys = ['maximum_iterations', 'error_on_nonconvergence', 'nonzero_initial_guess', 'absolute_tolerance', 'relative_tolerance']
        self.linear_dolfin_solver_settings = {**{k:v for k,v in self.linear_solver.__dict__.items() if k in linear_solver_keys}}
