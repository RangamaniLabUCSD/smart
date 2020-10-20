import pdb
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

class Config(object):
    """
    Refactored config 
         - directories
         - plot settings
         - reactions 
    """
    def __init__(self):

        # initialize with default values
        self.flags              = {'store_solutions': True}

        self.directory          = {'solutions': 'solutions',
                                   'plots': 'plots'}

        self.output_type        =  'xdmf'

        self.plot_settings      = {'lineopacity': 0.6,
                                   'linewidth_small': 0.6,
                                   'linewidth_med': 2.2,
                                   'fontsize_small': 3.5,
                                   'fontsize_med': 4.5,
                                   'figname': 'figure'}

        #self.probe_plot         = {'A': [(0.5,0.0), (1.0,0.0)]}
        self.probe_plot         = {}

        self.reaction_database  = {'prescribed': 'k',
                                   'prescribed_linear': 'k*u',
                                   'prescribed_leak': 'k*(1-u/umax)'}

    def check_config_validity(self):
        valid_filetypes = ['xdmf', 'vtk', None]
        if self.output_type not in valid_filetypes:
            raise ValueError(f"Only filetypes: '{valid_filetypes}' are supported.")

