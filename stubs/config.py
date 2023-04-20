"""
Configuration settings for simulation: plotting, reaction types, solution output, etc.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import dolfin as d
import ufl


__all__ = [
    "global_settings",
    "Config",
    "SolverConfig",
    "BaseConfig",
    "FlagsConfig",
    "LogLevelConfig",
]

_loglevel_to_int: Dict[str, int] = {
    "CRITICAL": int(d.LogLevel.CRITICAL),
    "ERROR": int(d.LogLevel.ERROR),
    "WARNING": int(d.LogLevel.WARNING),
    "INFO": int(d.LogLevel.INFO),
    "DEBUG": int(d.LogLevel.DEBUG),
    "NOTSET": 0,
}

global_settings = {
    "main_dir": None,
    "log_filename": None,
    # These functions will be substituted into any expressions
    "dolfin_expressions": {
        "exp": d.exp,
        "cos": d.cos,
        "sin": d.sin,
        "tan": d.tan,
        "cosh": ufl.cosh,
        "sinh": ufl.sinh,
        "tanh": ufl.tanh,
        "acos": d.acos,
        "asin": d.asin,
        "atan": d.atan,
        "atan2": ufl.atan_2,
        "sqrt": d.sqrt,
        "ln": d.ln,
        "abs": ufl.algebra.Abs,
        "sign": ufl.sign,
        "pi": d.pi,
        "erf": d.erf,
    },
}


class BaseConfig:
    """
    Base-class for setting configuration
    """

    def update(self, values: Dict[str, Any]):
        """
        Given a dictionary of input keys and values, update the named tuple.

        :throws AttributeError: If input key does not exist
        """
        for value, item in values.items():
            self.__setattr__(value, item)

    def __setattr__(self, key: str, value: Any):
        if key not in self.__annotations__:
            raise AttributeError(f"{key} not defined in {self.__annotations__.keys()}")
        else:
            object.__setattr__(self, key, value)

    def __getitem__(self, key):
        return self.__getattribute__(key)


@dataclass
class SolverConfig(BaseConfig):
    """
    Parameters for solver.

    :param final_t: End time of simulation
    :param use_snes: Use PETScSNES solver if true, else use DOLFINs NewtonSolver
    :param snes_preassemble_linear_system: If True separate linear components during assembly
    :param initial_dt: Initial time-stepping
    :param adjust_dt: A tuple (t, dt) of floats indicating when to next adjust the
        time-stepping and to what value
    :param dt: Number of digits for rounding `dt`
    :param print_assembly: Print information during assembly process
    :param dt_decrease_factor:
    :param dt_increase_factor:
    :param attempt_timestep_restart_on_divergence: Restart snes solver if it diverges
    """

    final_t: Optional[float] = None
    use_snes: bool = True
    snes_preassemble_linear_system: bool = (
        False  #: .. warning:: FIXME Currently untested
    )
    initial_dt: Optional[float] = None
    adjust_dt: Optional[Tuple[float, float]] = None
    time_precision: int = 6
    print_assembly: bool = True
    dt_decrease_factor: float = 1.0  #: .. warning:: FIXME Currently unused parameter
    dt_increase_factor: float = 1.0  #: .. warning:: FIXME Currently unused parameter
    attempt_timestep_restart_on_divergence: bool = False  # testing in progress
    reset_timestep_for_negative_solution: bool = False


@dataclass
class FlagsConfig(BaseConfig):
    """
    Various flags

    :param store_solutions: Store solutions to file.
    :param allow_unused_components: Allow parameters not defined in any reaction to be
        defined in any model.
    :param print_verbose_info: Print detailed information about a model
    """

    store_solutions: bool = True
    allow_unused_components: bool = False
    print_verbose_info: bool = True


@dataclass
class LogLevelConfig(BaseConfig):
    """
    Settings for logging

    :param FFC: LogLevel for FFC
    :param UFL: LogLevel for UFL
    :param dijitso: LogLevel for dijitso
    :param dolfin: LogLevel for dolfin

    """

    FFC: str = "DEBUG"
    UFL: str = "DEBUG"
    djitso: str = "DEBUG"
    dolfin: str = "INFO"

    def set_logger_levels(self):
        """
        For each of the loggers, set the appropriate log-level
        """
        # set for dolfin
        d.set_log_level(_loglevel_to_int[self.dolfin])

        # set for others
        other_loggers = list(self.__annotations__)
        print(other_loggers)
        other_loggers.remove("dolfin")
        for logger_name in other_loggers:
            logging.getLogger(logger_name).setLevel(
                _loglevel_to_int[self.__getattribute__(logger_name)]
            )


@dataclass
class Config:
    """
    Configuration settings.

    :param solver: Options for the solvers
    :param flags: Various options
    :param directory: Outputting options
    :param loglevel: Logging options for FEniCS modules
    """

    solver: SolverConfig = field(default_factory=SolverConfig)
    flags: FlagsConfig = field(default_factory=FlagsConfig)
    loglevel: LogLevelConfig = field(default_factory=LogLevelConfig)

    @property
    def reaction_database(self) -> Dict[str, str]:
        """
        Return database of known reactions
        """
        return {"prescribed": "k", "prescribed_linear": "k*u"}

    def set_logger_levels(self):
        self.loglevel.set_logger_levels()

    def set_all_logger_levels(self, log_level: int):
        """
        Set all loggers to a given loglevel
        :param log_level: The loglevel: `(0,10,20,30,40,50)`
        """
        # Update LogLevel class
        for logger_name in self.loglevel.__annotations__:
            self.loglevel.__setattr__(logger_name, log_level)
        # Set LogLevels to logger
        self.set_logger_levels()
