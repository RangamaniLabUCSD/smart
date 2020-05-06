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

class ConfigRefactor(object):
    """
    Refactored config 
         - directories
         - mesh
         - plot settings
    """
    def __init__(self, mesh_filename=None):
        self.mesh_filename      = mesh_filename

        # initialize with default values

        self.directory          = {'parent': 'results',
                                   'solutions': 'solutions',
                                   'plots': 'plots',
                                   'relative': True}

        self.output_type        =  'xdmf'

        self.plot_settings      = {'lineopacity': 0.6,
                                   'linewidth_small': 0.6,
                                   'linewidth_med': 2.2,
                                   'fontsize_small': 3.5,
                                   'fontsize_med': 4.5,
                                   'figname': 'figure'}

        self.probe_plot         = {'species': [],
                                   'coords': [(0.5, 0.5, 0.5)]}

        self.reaction_database  = {'prescribed': 'k',
                                   'prescribed_linear': 'k*u',
                                   'prescribed_leak': 'k*(1-u/umax)'}

    def check_config_validity(self):
        if self.mesh_filename is None:
            raise TypeError("mesh_filename must be provided in the configuration file.")

    def generate_model(self):

        if not all([x in self.model.keys() for x in ['parameters', 'species', 'compartments', 'reactions']]):
            raise Exception("Parameters, species, compartments, and reactions must all be specified.")
        PD = self._json_to_ObjectContainer(self.model['parameters'], 'parameters')
        SD = self._json_to_ObjectContainer(self.model['species'], 'species')
        CD = self._json_to_ObjectContainer(self.model['compartments'], 'compartments')
        RD = self._json_to_ObjectContainer(self.model['reactions'], 'reactions')

        # parameter/unit assembly
        PD.do_to_all('assemble_units', {'unit_name': 'unit'})
        PD.do_to_all('assemble_units', {'value_name':'value', 'unit_name':'unit', 'assembled_name': 'value_unit'})
        PD.do_to_all('assembleTimeDependentParameters')
        SD.do_to_all('assemble_units', {'unit_name': 'concentration_units'})
        SD.do_to_all('assemble_units', {'unit_name': 'D_units'})
        CD.do_to_all('assemble_units', {'unit_name':'compartment_units'})
        RD.do_to_all('initialize_flux_equations_for_known_reactions', {"reaction_database": self.reaction_database})


        # linking containers with one another
        RD.link_object(PD,'paramDict','name','paramDictValues', value_is_key=True)
        SD.link_object(CD,'compartment_name','name','compartment')
        SD.copy_linked_property('compartment', 'dimensionality', 'dimensionality')
        RD.do_to_all('get_involved_species_and_compartments', {"SD": SD, "CD": CD})
        RD.link_object(SD,'involved_species','name','involved_species_link')
        #RD.do_to_all('combineDicts', {'dict1': 'paramDictValues', 'dict2': 'involved_species_link', 'new_dict_name': 'varDict'})

        # meshes
        CD.add_property('meshes', self.mesh)
        CD.load_mesh('cyto', self.mesh['cyto'])
        CD.extract_submeshes('cyto', False)
        CD.compute_scaling_factors()

        num_species_per_compartment = RD.get_species_compartment_counts(SD, CD, self.settings)
        CD.get_min_max_dim()
        SD.assemble_compartment_indices(RD, CD, self.settings)
        CD.add_property_to_all('is_in_a_reaction', False)
        CD.add_property_to_all('V', None)

        #RD.replace_sub_species_in_reactions(SD)
        #CD.Print()

        # # # dolfin
        SD.assemble_dolfin_functions(RD, CD, self.settings)
        SD.assign_initial_conditions()

        RD.reaction_to_fluxes()
        RD.do_to_all('reaction_to_fluxes')
        FD = RD.get_flux_container()
        FD.do_to_all('get_additional_flux_properties', {"CD": CD, "config": self})

        # # opportunity to make custom changes

        FD.do_to_all('flux_to_dolfin', {"config": self})
        FD.check_and_replace_sub_species(SD, CD, self)

        model = model_assembly.Model(PD, SD, CD, RD, FD, self)

        # to deal with possible floating point error in mesh coordinates
        model.set_allow_extrapolation()
        # Turn fluxes into fenics/dolfin expressions
        model.assemble_reactive_fluxes()
        model.assemble_diffusive_fluxes()
        #model.establish_mappings()


        # # debug
        # model.PD.print()
        # model.SD.print()
        # model.CD.print()
        # model.RD.print()
        # model.FD.print()
        # return model

        # Sort forms by type (diffusive, time derivative, etc.)
        model.sort_forms()

        if rank==root:
            Print("Model created succesfully! :)")
            model.PD.print()
            model.SD.print()
            model.CD.print()
            model.RD.print()
            model.FD.print()

        return model

