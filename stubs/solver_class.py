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


class Setting(object):
    def __init__(self, value, type_, required='False', validity_condition='True'):
        """
        required can either be True, False, or conditional on another setting;
        in the last case required is a dictionary where key/value is the
        setting and its value that must be true)
        validity_condition: str 
        
        Args:
            value (variable): Value of setting
            type_ (variable): Type of setting (if this differs from type
            (setting.value) an Exception will be raised)
            required (str, optional): str that evaluates to a lambda function
            with input argument 'solver'
            validity_condition (str, optional): str that evaluates to a lambda
            function with input argument 'value'
        """

        # example: setting 'b' has initial value 0.0, type_ 'float', is only a
        # required setting if setting 'a' is equal to True, and must be >= 0. 
        # Initialize as such:
        # 
        # Setting(0.0, 'b', float, required=lambda s: s.a.value==True, 
        # validity_condition=lambda val: val>=0)
        self.value              = value
        self.type_              = type_
        self.required           = required
        self.validity_condition = validity_condition

        self._required_lambda           = self.string_to_lambda(required, 'solver')
        self._validity_condition_lambda = self.string_to_lambda(validity_condition, 'value')

    def string_to_lambda(self, string, var_name):
        """
        Converts a string representing a lambda function with a single input
        variable into a lambda function
        """
        if type(string) is not str or type(var_name) is not str:
            raise Exception(f"string and var_name must both have type str")

        return eval('lambda ' + var_name + ': ' + string)

class Solver(object):
    def __init__(self, method):
        self.method = Setting(None, str, required='True')
    def check_validity(self, verbose=False):
        """
        Check that all current settings are valid
        """
        for setting_name, setting in self.__dict__.items():
            if type(setting) == Setting:
                # type checks
                if type(setting.value) != setting.type_:
                    raise Exception(f"The type of setting \"{setting_name}\" is \"{type(setting.value)}\". \
                                      Required type is \"{setting.type_}\"")

                # validity condition check
                if setting._validity_condition_lambda(setting.value) not in (True, False):
                    raise Exception(f"The validity condition for setting \"{setting_name}\" must return either True or False")
                if setting._validity_condition_lambda(setting.value) is False:
                    raise Exception(f"The validity condition for setting \"{setting_name}\" was not satisfied.")

                # requirement/dependency check
                if setting._required_lambda(self) not in (True, False):
                    raise Exception(f"The valdiity condition for setting \"{setting_name}\" must return either True or False")
                if setting._required_lambda(self) is False:
                    print(f"Setting \"{setting_name}\" was provided but is not required.")
                elif setting._required_lambda(self) is True:
                    if setting.value is None:
                        raise Exception(f"Required setting \"{setting_name}\" has not been initialized!")

                # print a summary
                if verbose is True:
                    print(f"Setting \"{setting_name}\" passes all checks and is valid:")
                    print(f"\t* Type of setting, {type(setting.value)} matches required type, {setting.type_.__name__}")
                    print(f"\t* Validity condition, {setting.validity_condition}, is satisfied.")

        print(f"All settings for \"{self.__class__.__name__}\" are valid!")


class MultiphysicsSolver(Solver):
    def __init__(self, method='iterative'):
        super().__init__(method)
        # default values
        self.method.value               = method
        self.method.validity_condition  = "value in ['iterative', 'monolithic']"

        self.iteration_tol = Setting(1e-6, float, required="solver.method.value=='iterative'",
                                     validity_condition="value >= 0.0")

# example call
mps = MultiphysicsSolver('iterative')
# check validity
mps.check_validity(verbose=True)

class NonlinearSolver(Solver):
    def __init__(self, method='newton'):
        super().__init__(method)
        # default values
        self.method.value               = method
        self.method.validity_condition  = "x in ['newton', 'picard']"

        self.min_nonlinear      = Setting(2, int, required="True")
        self.max_nonlinear      = Setting(10, int, required="True")
        self.dt_increase_factor = Setting(1.0, float, required="True")
        self.dt_decrease_factor = Setting(0.8, float, required="True")
        self.dolfin_newton      = Setting({'maximum_iterations': 50, 'error_on_nonconvergence': False,
                                           'relative_tolerance': 1e-8, 'absolute_tolerance': 1e-10},
                                           dict, required="solver.method.value=='newton'")

        self.picard_norm        = Setting('Linf', str, required="solver.method.value=='picard'",
                                          validity_condition="value in ['Linf', 'L2']")
        self.relative_tolerance = Setting(1e-8, float, required="solver.method.value=='picard'")
        self.absolute_tolerance = Setting(1e-8, float, required="solver.method.value=='picard'")

nls = NonlinearSolver('newton')
nls.check_validity(verbose=True)

class NonLinearPicardSolver(NonLinearSovler):

    valid_norms = ['Linf', 'L2']

    def __init__(self, mix_nonlinear=2, max_nonlinear=10, dt_increase_factor=1.0, dt_decrease_factor=0.8,
                 picard_norm = 'Linf', ...
                 ):
        super().__init__(max_nonlinear...)

        # EXPENSIVE!

        if picard_norm not in valid_norms:
            raise RuntimeException(f"Invalid Picard norm ({picard_norm}) reasonable values are: {valid_norms}")
        self.picard_norm = picard_norm


class LinearSolver(Solver):
    def __init__(self, method):
        super().__init__(Solver, method)


            {
            'solver':
                {
                'backend': 'dolfin',
                'multiphysics':
                    {
                    'scheme': 'iterative', # monolithic
                    'iteration_tol': 1e-6,
                    },
                'nonlinear':
                    {
                    'scheme': 'newton', # also, 'picard'
                    # min/max number of non-linear iterations to determine if we should increase/decrease dt
                    'min_nonlinear': 2, 
                    'max_nonlinear': 10,
                    'dt_increase_factor': 1.0,
                    'dt_decrease_factor': 0.7,
                    'dolfin_newton':
                        {
                        'maximum_iterations': 50,
                        'error_on_nonconvergence': False,
                        'relative_tolerance': 1e-8,
                        'absolute_tolerance': 1e-10,
                        },
                    'picard': 
                        {
                        'norm': 'Linf',
                        'relative_tolerance': 1e-8,
                        'absolute_tolerance': 1e-10,
                        },
                    },
                'linear':
                    {
                    'dolfin_krylov': # gets put directly into krylov solver parameter settings
                        {
                        'maximum_iterations': 100000,
                        'error_on_nonconvergence': False,
                        'nonzero_initial_guess': True,
                        'relative_tolerance': 1e-8,
                        'absolute_tolerance': 1e-10,
                        },
                    }

                },
            'ignore_surface_diffusion': False, # setting to True will treat system as ODEs 
            'auto_preintegrate': True,
            'reaction_database':
                {
                'prescribed': 'k',
                'prescribed_linear': 'k*u',
                'leak': 'k*(1-u/umax)',
                'hillI': 'k*u**n/(u**n+K**n)',
                },
            'plot':
                {
                'filetype': 'xdmf',
                'lineopacity': 0.6,
                'linewidth_small': 0.6,
                'linewidth_med': 2.2,
                'fontsize_small': 3.5,
                'fontsize_med': 4.5,
                'figname': 'concentration_plot',
                },
            }

        # user must provide these settings
        self._required_settings = [
            'save_directory', # directory to save solutions/plots to
            'mesh', # primary mesh
            #'model', # model file
            'initial_dt', 
            'T' # final time of simulation
            ]
        self._optional_settings = [
            'reset_times',
            'reset_dt',
            'output',
        ]
#$ advanced.reset_times = [0.0010, 0.0100, 0.0110, 0.0200, 0.0210, 0.0300, 0.0310, 0.0400, 0.0410, 0.05, 0.0996, 0.10000, 0.110, 0.150, 0.200]
#$ advanced.reset_dt    = [0.0030, 0.0003, 0.0030, 0.0003, 0.0030, 0.0003, 0.0030, 0.0003, 0.0030, 0.01, 0.0001, 0.00002, 0.002, 0.025, 0.050]
#$ output.species = [A]
#$ output.points_x = [0.5]
#$ output.points_y = [0.5]
#$ output.points_z = [0.5]


    def check_config_validity(self):
        """
        Checks that the config file is valid
        """
        # Check that all required settings have been provided
        for setting in self._required_settings:
            if setting not in self.keys():
                raise Exception(f"Required setting, {setting}, was not found in config.")
        # Check that the times are sensical
        if self['initial_dt'] <= 0.0:
            raise Exception(f"Initial time-step must be larger than 0.0")

        print("Config file is valid")


    def to_json(self, filename):
        """
        Writes out settings to a json file
        """
        with open(filename, 'w') as outfile:
            outfile.write(json.dumps(self.settings, indent=4))
    def from_json(self, filename):
        """
        Reads in settings from a json file
        """
        with open(filename, 'r') as infile:
            self.settings =  json.load(infile)



            
        #======= Move to Model()
        if 'directory' in self.model.keys():
            model_dir = self.model['directory']
            Print("\nAssuming file names, loading from directory %s" % model_dir)
            self.model['parameters'] = model_dir + 'parameters.json'
            self.model['compartments'] = model_dir + 'compartments.json'
            self.model['species'] = model_dir + 'species.json'
            self.model['reactions'] = model_dir + 'reactions.json'

        if (all([x in self.model.keys() for x in ['parameters', 'species', 'compartments', 'reactions']])
            and self.mesh.keys()):
            Print("Parameters, species, compartments, reactions, and a mesh were imported succesfully!")

        file.close()

    def find_mesh_midpoint(self,filename):
        m = d.Mesh(filename)
        return (m.coordinates().max(axis=0) - m.coordinates().min(axis=0))/2

    def find_mesh_COM(self,filename):
        m = d.Mesh(filename)
        return m.coordinates().mean(axis=0)





    def generate_model(self):
        """
        Processes the parameters, species, compartments, and reactions before
        equations are generated.
        """

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
        RD.link_object(PD,'param_dict','name','param_dict_values', value_is_key=True)
        SD.link_object(CD,'compartment_name','name','compartment')
        SD.copy_linked_property('compartment', 'dimensionality', 'dimensionality')
        RD.do_to_all('get_involved_species_and_compartments', {"SD": SD, "CD": CD})
        RD.link_object(SD,'involved_species','name','involved_species_link')
        #RD.do_to_all('combineDicts', {'dict1': 'param_dict_values', 'dict2': 'involved_species_link', 'new_dict_name': 'varDict'})

        # meshes
        CD.add_property('meshes', self.mesh)
        #for mesh_name in self.mesh.keys():
        CD.load_mesh('cyto', self.mesh['cyto'])
        CD.extract_submeshes('cyto', False)
        #CD.load_mesh('cyto', self.mesh['cyto'])
        #CD.extract_submeshes('cyto', False)
        CD.compute_scaling_factors()

        num_species_per_compartment = RD.get_species_compartment_counts(SD, CD, self.settings)
        CD.get_min_max_dim()
        SD.assemble_compartment_indices(RD, CD, self.settings)
        CD.add_property_to_all('is_in_a_reaction', False)
        CD.add_property_to_all('V', None)

        #RD.replace_sub_species_in_reactions(SD)
        #CD.Print()

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
        # Sort forms by type (diffusive, time derivative, etc.)
        model.sort_forms()

        model.FD.print()

        return model
        if rank==root:
            Print("Model created succesfully! :)")
            model.PD.print()
            model.SD.print()
            model.CD.print()
            model.RD.print()
        return model

        # ====== From here on out add to Solver class



    def _json_to_ObjectContainer(self, json_file_name, data_type=None):
        if not data_type:
            raise Exception("Please include the type of data this is (parameters, species, compartments, reactions).")
        if not os.path.exists(json_file_name):
            raise Exception("Cannot find JSON file, %s"%json_file_name)
        df = read_json(json_file_name).sort_index()
        df = nan_to_none(df)
        if data_type in ['parameters', 'parameter', 'param', 'p']:
            return model_assembly.ParameterContainer(df)
        elif data_type in ['species', 'sp', 'spec', 's']:
            return model_assembly.SpeciesContainer(df)
        elif data_type in ['compartments', 'compartment', 'comp', 'c']:
            return model_assembly.CompartmentContainer(df)
        elif data_type in ['reactions', 'reaction', 'r', 'rxn']:
            return model_assembly.ReactionContainer(df)
        else:
            raise Exception("I don't know what kind of ObjectContainer this .json file should be")
