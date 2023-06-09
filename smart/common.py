"""
General functions: array manipulation, data i/o, etc
"""

import time
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Union

import dolfin as d

import ufl
from pytz import timezone
from termcolor import colored

from .deprecation import deprecated
from .config import global_settings as gset, FancyFormatter
from .units import unit

__all__ = [
    "smart_expressions",
    "sub",
    "Stopwatch",
]


comm = d.MPI.comm_world
rank = comm.rank
size = comm.size
root = 0

logger = logging.getLogger(__name__)


def smart_expressions(
    dolfin_expressions: Dict[str, Callable[[Any], Any]]
) -> Dict[str, Callable[[Any], Union[ufl.core.expr.Expr, float]]]:
    """
    Map strings to DOLFIN/UFL functions, that takes in
    `smart`-Expressions, i.e. functions with a unit.

    Args:
        dolfin_expressions: Dictionary of strings mapping
          to smart expressions

    Example:

        .. highlight:: python
        .. code-block:: python

            input = {"sin": ufl.sin}
            output = smart_expressions(input)

        Output is then a dictionary that maps "sin" to
        a function :code:`sin(x)` that takes in a
        function with a unit and returns
        :code:`ufl.sin(x.to(unit.dimensionless).magnitude)`
    """
    # must use a function here or else all functions are defined as the final "v"
    # see e.g. https://stackoverflow.com/questions/36805071
    def dict_entry(v):
        return lambda x: v(x.to(unit.dimensionless).magnitude)

    return {k: dict_entry(v) for k, v in dolfin_expressions.items()}


def sub(
    func: Union[
        List[Union[d.Function, d.FunctionSpace]],
        d.Function,
        d.MixedFunctionSpace,
        d.FunctionSpace,
        d.function.argument.Argument,
    ],
    idx: int,
    collapse_function_space: bool = True,
):
    """
    A utility function for getting the sub component of a

    - Functionspace
    - Mixed function space
    - Function
    - Test/Trial Function

    Args:
        func: The input
        idx: The component to extract
        collapse_function_space: If the input `func`
        is a `dolfin.FunctionSpace`, collapse it if `True`.

    Returns:
        The relevant component of the input
    """

    if isinstance(func, (list, tuple)):
        return func[idx]

    elif isinstance(func, d.Function):
        # MixedFunctionSpace
        if func._functions:
            assert func.sub(idx) is func._functions[idx]
            return func.sub(idx)
        else:
            # scalar FunctionSpace
            if func.num_sub_spaces() <= 1 and idx == 0:
                return func
            else:
                # Use d.split() to get subfunctions from a
                # VectorFunctionSpace that can be used in forms
                return func.sub(idx)

    elif isinstance(func, d.MixedFunctionSpace):
        return func.sub(idx)

    elif isinstance(func, d.FunctionSpace):
        if func.num_sub_spaces() <= 1 and idx == 0:
            return func
        else:
            if collapse_function_space:
                return func.sub(idx).collapse()
            else:
                return func.sub(idx)

    elif isinstance(func, d.function.argument.Argument):
        if func.function_space().num_sub_spaces() <= 1 and idx == 0:
            return func
        else:
            return func[idx]

    else:
        raise ValueError(f"Unknown input type of {func=}")


# Write a stopwatch class to measure time elapsed
# with a start, stop, and pause methods
# Keep track of timings in a list of lists called
# self.timings, each time the timer is paused,
# the time elapsed since the last pause is added to
# the sublist. Using stop resets the timer to zero
# and beings a new list of timings.


class Stopwatch:
    "Basic stopwatch class with inner/outer timings (pause and stop)"

    def __init__(self, name=None, time_unit="s", print_buffer=0, filename=None, start=False):
        self.name = name
        self.time_unit = time_unit
        self.stop_timings = []  # length = number of stops
        self.pause_timings = []  # length = number of stops (list of lists)
        self._pause_timings = []  # length = number of pauses (reset on stop)
        self._times = []
        self.is_paused = True
        self.print_buffer = print_buffer
        self._print_name = f"{str(self.name): <{self.print_buffer}}"
        # self.start()
        self.filename = filename
        if filename is not None:
            self.file_logger = logging.getLogger("smart_stop_watch")
            handler = logging.FileHandler(filename=filename)
            handler.setFormatter(FancyFormatter())
            self.file_logger.addHandler(handler)
            self.file_logger.setLevel(logging.DEBUG)
        else:
            # Just use the regular logger
            self.file_logger = logger
        if start:
            self.start()

    def start(self):
        self._times.append(time.time())
        self.is_paused = False

    def pause(self):
        if self.is_paused:
            return
        else:
            self._times.append(time.time())
            self._pause_timings.append(self._times[-1] - self._times[-2])
            self.is_paused = True
            self.file_logger.debug(
                f"{self.name} (iter {len(self._pause_timings)}) finished "
                f"in {self.time_str(self._pause_timings[-1])} {self.time_unit}",
                extra=dict(format_type="logred"),
            )

    def stop(self, print_result=True):
        self._times.append(time.time())
        if self.is_paused:
            final_time = 0
        else:
            final_time = self._times[-1] - self._times[-2]
            self.is_paused = True
        total_time = sum(self._pause_timings) + final_time
        self.stop_timings.append(total_time)
        if print_result:
            self.file_logger.debug(
                f"{self._print_name} finished in {self.time_str(total_time)} {self.time_unit}",
                extra=dict(format_type="logred"),
            )

        self.pause_timings.append(self._pause_timings)
        self._pause_timings = []
        self._times = []

    def set_timing(self, timing):
        self.stop_timings.append(timing)
        self.file_logger.debug(
            f"{self._print_name} finished in {self.time_str(timing)} {self.time_unit}",
            extra=dict(format_type="logred"),
        )

    def print_last_stop(self):
        self.file_logger.debug(
            f"{self._print_name} finished in "
            f"{self.time_str(self.stop_timings[-1])} {self.time_unit}",
            extra=dict(format_type="logred"),
        )

    def time_str(self, t):
        return str({"us": 1e6, "ms": 1e3, "s": 1, "min": 1 / 60}[self.time_unit] * t)[0:8]


@deprecated
def _fancy_print(
    title_text,
    buffer_color=None,
    text_color=None,
    filler_char=None,
    num_banners=None,
    new_lines=None,
    left_justify=None,
    format_type="default",
    include_timestamp=True,
    filename=None,
):
    "Formatted text to stand out."

    # Initialize with the default options
    buffer_color_ = "cyan"
    text_color_ = "green"
    filler_char_ = "="
    num_banners_ = 0
    new_lines_ = [0, 0]
    left_justify_ = False
    # Override with format_type options
    if format_type == "default":
        pass
    elif format_type == "title":
        text_color_ = "magenta"
        num_banners_ = 1
        new_lines_ = [1, 0]
    elif format_type == "subtitle":
        text_color_ = "green"
        filler_char_ = "."
        left_justify_ = True
    elif format_type == "data":
        buffer_color_ = "white"
        text_color_ = "white"
        filler_char_ = ""
        left_justify_ = True
    elif format_type == "data_important":
        buffer_color_ = "white"
        text_color_ = "red"
        filler_char_ = ""
        left_justify_ = True
    elif format_type == "log":
        buffer_color_ = "white"
        text_color_ = "green"
        filler_char_ = ""
        left_justify_ = True
    elif format_type == "logred":
        buffer_color_ = "white"
        text_color_ = "red"
        filler_char_ = ""
        left_justify_ = True
    elif format_type == "log_important":
        buffer_color_ = "white"
        text_color_ = "magenta"
        filler_char_ = "."
    elif format_type == "log_urgent":
        buffer_color_ = "white"
        text_color_ = "red"
        filler_char_ = "."
    elif format_type == "warning":
        buffer_color_ = "magenta"
        text_color_ = "red"
        filler_char_ = "!"
        num_banners_ = 2
        new_lines_ = [1, 1]
    elif format_type == "timestep":
        text_color_ = "red"
        num_banners_ = 2
        filler_char_ = "."
        new_lines_ = [1, 1]
    elif format_type == "solverstep":
        text_color_ = "red"
        num_banners_ = 1
        filler_char_ = "."
        new_lines_ = [1, 1]
    elif format_type == "assembly":
        text_color_ = "magenta"
        num_banners_ = 0
        filler_char_ = "."
        new_lines_ = [1, 0]
    elif format_type == "assembly_sub":
        text_color_ = "magenta"
        num_banners_ = 0
        filler_char_ = ""
        new_lines_ = [0, 0]
        left_justify_ = True
    elif format_type is not None:
        raise ValueError("Unknown formatting_type.")

    # Override again with user options
    if buffer_color is None:
        buffer_color = buffer_color_
    if text_color is None:
        text_color = text_color_
    if filler_char is None:
        filler_char = filler_char_
    if num_banners is None:
        num_banners = num_banners_
    if new_lines is None:
        new_lines = new_lines_
    if left_justify is None:
        left_justify = left_justify_

    # include MPI rank in message
    if size > 1:
        title_text = f"CPU {rank}: {title_text}"
    if include_timestamp:
        timestamp = datetime.now(timezone("US/Pacific")).strftime("[%Y-%m-%d time=%H:%M:%S]")
        title_text = f"{timestamp} {title_text}"

    # calculate optimal buffer size
    min_buffer_size = 5
    terminal_width = 120
    buffer_size = max(
        [min_buffer_size, int((terminal_width - 1 - len(title_text)) / 2 - 1)]
    )  # terminal width == 80
    title_str_len = (buffer_size + 1) * 2 + len(title_text)
    parity = 1 if title_str_len == 78 else 0

    # color/stylize buffer, text, and banner
    def buffer(buffer_size):
        return colored(filler_char * buffer_size, buffer_color)

    if left_justify:
        title_str = f"{colored(title_text, text_color)} {buffer(buffer_size*2+1+parity)}"
    else:
        title_str = (
            f"{buffer(buffer_size)} {colored(title_text, text_color)} "
            f"{buffer(buffer_size+parity)}"
        )
    banner = colored(filler_char * (title_str_len + parity), buffer_color)

    def print_out(text, filename=None):
        "print to file and terminal"
        if filename is not None:
            with open(filename, "a") as f:
                f.write(text + "\n")
        elif gset["log_filename"] is not None:
            with open(gset["log_filename"], "a") as f:
                f.write(text + "\n")
        print(text)

    # initial spacing
    if new_lines[0] > 0:
        print_out("\n" * (new_lines[0] - 1), filename)
    # print first banner
    for _ in range(num_banners):
        print_out(f"{banner}", filename)
    # print main text
    print_out(title_str, filename)
    # print second banner
    for _ in range(num_banners):
        print_out(f"{banner}", filename)
    # end spacing
    if new_lines[1] > 0:
        print_out("\n" * (new_lines[1] - 1), filename)
