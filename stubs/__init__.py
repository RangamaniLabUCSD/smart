"""
stubs
Fenics stuff
"""
import dolfin as d 
import pandas as pd 
import sympy
import pickle
from sympy.parsing.sympy_parser import parse_expr
from termcolor import colored
import pint
unit = pint.UnitRegistry()
unit.define('molecule = mol/6.022140857e23')

# Add imports here
from stubs import *
from . import model_building
from .stubs import *
from . import data_manipulation
from . import flux_assembly
from . import model_assembly
from . import model_building
from . import solvers


# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
