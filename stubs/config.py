"""
Configuration settings for simulation: plotting, reaction types, solution output, etc.
"""
import logging
from logging import config as logging_config
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple, NamedTuple


from termcolor import colored
import dolfin as d
import ufl


__all__ = [
    "global_settings",
    "Config",
    "SolverConfig",
    "BaseConfig",
    "FlagsConfig",
]


comm = d.MPI.comm_world
rank = comm.rank
size = comm.size
root = 0
fancy_format = (
    "%(asctime)s %(rank)s%(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
)
base_format = "%(asctime)s %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
root_format = "ROOT -" + base_format


class FormatType(str, Enum):
    default = "default"
    title = "title"  # type: ignore
    subtitle = "subtitle"
    data = "data"
    data_important = "data_important"
    log = "log"
    logred = "logred"
    log_important = "log_important"
    log_urgent = "log_urgent"
    warning = "warning"
    timestep = "timestep"
    solverstep = "solverstep"
    assembly = "assembly"
    assembly_sub = "assembly_sub"


class FormatOption(NamedTuple):
    buffer_color: str = "cyan"
    text_color: str = "white"
    filler_char: str = ""
    num_banners: int = 0
    new_lines: Tuple[int, int] = (0, 0)
    left_justify: bool = False


def format_type_to_options(format_type: FormatType) -> FormatOption:
    """Take a format type and return the corresponding options

    Parameters
    ----------
    format_type : FormatType
        The format type

    Returns
    -------
    FormatOption
        The options

    Raises
    ------
    ValueError
        If provided format type does not exist.
    """
    if FormatType[format_type] == FormatType.default:
        return FormatOption()
    elif FormatType[format_type] == FormatType.title:
        return FormatOption(
            text_color="magenta",
            num_banners=1,
            new_lines=(1, 0),
        )
    elif FormatType[format_type] == FormatType.subtitle:
        return FormatOption(
            text_color="green",
            filler_char=".",
            left_justify=True,
        )
    elif FormatType[format_type] == FormatType.data:
        return FormatOption(
            buffer_color="white",
            text_color="white",
            filler_char="",
            left_justify=True,
        )
    elif FormatType[format_type] == FormatType.data_important:
        return FormatOption(
            buffer_color="white",
            text_color="red",
            filler_char="",
            left_justify=True,
        )
    elif FormatType[format_type] == FormatType.log:
        return FormatOption(
            buffer_color="white",
            text_color="green",
            filler_char="",
            left_justify=True,
        )
    elif FormatType[format_type] == FormatType.logred:
        return FormatOption(
            buffer_color="white",
            text_color="red",
            filler_char="",
            left_justify=True,
        )
    elif FormatType[format_type] == FormatType.log_important:
        return FormatOption(
            buffer_color="white",
            text_color="magenta",
            filler_char=".",
        )
    elif FormatType[format_type] == FormatType.log_urgent:
        return FormatOption(
            buffer_color="white",
            text_color="red",
            filler_char=".",
        )
    elif FormatType[format_type] == FormatType.warning:
        return FormatOption(
            buffer_color="magenta",
            text_color="red",
            filler_char="!",
            num_banners=2,
            new_lines=(1, 1),
        )
    elif FormatType[format_type] == FormatType.timestep:
        return FormatOption(
            text_color="red",
            num_banners=2,
            filler_char=".",
            new_lines=(1, 1),
        )
    elif FormatType[format_type] == FormatType.solverstep:
        return FormatOption(
            text_color="red",
            num_banners=1,
            filler_char=".",
            new_lines=(1, 1),
        )
    elif FormatType[format_type] == FormatType.assembly:
        return FormatOption(
            text_color="magenta",
            num_banners=0,
            filler_char=".",
            new_lines=(1, 0),
        )
    elif FormatType[format_type] == FormatType.assembly_sub:
        return FormatOption(
            text_color="magenta",
            num_banners=0,
            filler_char="",
            new_lines=(0, 0),
            left_justify=True,
        )
    else:
        msg = f"Invalid format type {format_type}"
        raise ValueError(msg)


def format_message(title_text: str, format_option: FormatOption) -> Tuple[str, str]:
    """Format message according to the format options

    Parameters
    ----------
    title_text : str
        The message to be formatted
    format_option : FormatOption
        The options for formatting

    Returns
    -------
    Tuple[str, str]
        (banners, formatted text)
    """
    min_buffer_size = 5
    terminal_width = 120
    buffer_size = max(
        [min_buffer_size, int((terminal_width - 1 - len(title_text)) / 2 - 1)]
    )  # terminal width == 80
    title_str_len = (buffer_size + 1) * 2 + len(title_text)
    parity = 1 if title_str_len == 78 else 0

    # color/stylize buffer, text, and banner
    def buffer(buffer_size):
        return colored(format_option.filler_char * buffer_size, format_option.buffer_color)

    if format_option.left_justify:
        title_str = (
            f"{colored(title_text, format_option.text_color)} {buffer(buffer_size*2+1+parity)}"
        )
    else:
        title_str = (
            f"{buffer(buffer_size)} {colored(title_text, format_option.text_color)} "
            f"{buffer(buffer_size+parity)}"
        )
    banner = colored(
        format_option.filler_char * (title_str_len + parity), format_option.buffer_color
    )
    return banner, title_str


class FancyFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # Set the rank attribute with is a parameter in the `fancy_format``
        record.rank = f"CPU {rank}: " if size > 1 else ""

        # Extract the format options. Here we expect that `format_type` is provided
        # as part of the `extra` dictionary passed to the logger
        format_option = format_type_to_options(getattr(record, "format_type", FormatType.default))

        # Create the formatter
        formatter = logging.Formatter(fancy_format)
        # Format the record
        out = formatter.format(record)
        # Now create banners and formatted text
        banner, formatted_out = format_message(out, format_option=format_option)

        # Finally construct the output with banners and new lines
        banners = f"\n{banner}" * format_option.num_banners + "\n"
        prefix = "\n" * format_option.new_lines[0]
        postfix = ""
        if format_option.num_banners > 0:
            prefix += banners
            postfix += banners
        postfix += "\n" * format_option.new_lines[1]
        return prefix + formatted_out + postfix


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "root": {"format": root_format},
        "base": {"format": base_format},
        "default": {"()": FancyFormatter},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "default"},
        "root_console": {"class": "logging.StreamHandler", "formatter": "root"},
    },
    "loggers": {
        "stubs": {
            "handlers": ["console"],
            "level": "DEBUG",
            # Don't send it up my namespace for additional handling
            "propagate": False,
        },
        "pint": {"level": "ERROR"},
        "FFC": {"level": "WARNING"},
        "UFL": {"level": "WARNING"},
        "dolfin": {"level": "INFO"},
        "dijitso": {"level": "INFO"},
    },
    "root": {"handlers": ["root_console"], "level": "INFO"},
}


logging_config.dictConfig(LOGGING_CONFIG)


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
    snes_preassemble_linear_system: bool = False  #: .. warning:: FIXME Currently untested
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

    @property
    def reaction_database(self) -> Dict[str, str]:
        """
        Return database of known reactions
        """
        return {"prescribed": "k", "prescribed_linear": "k*u"}
