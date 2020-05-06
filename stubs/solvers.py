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



# class Setting(object):
#     def __init__(self, value, type_, required='False', validity_condition='True'):
#         """
#         required can either be True, False, or conditional on another setting;
#         in the last case required is a dictionary where key/value is the
#         setting and its value that must be true)
#         validity_condition: str 
        
#         Args:
#             value (variable): Value of setting
#             type_ (variable): Type of setting (if this differs from type
#             (setting.value) an Exception will be raised)
#             required (str, optional): str that evaluates to a lambda function
#             with input argument 'solver'
#             validity_condition (str, optional): str that evaluates to a lambda
#             function with input argument 'value'
#         """

#         # example: setting 'b' has initial value 0.0, type_ 'float', is only a
#         # required setting if setting 'a' is equal to True, and must be >= 0. 
#         # Initialize as such:
#         # 
#         # Setting(0.0, 'b', float, required=lambda s: s.a.value==True, 
#         # validity_condition=lambda val: val>=0)
#         self.value              = value
#         self.type_              = type_
#         self.required           = required
#         self.validity_condition = validity_condition

#         self._required_lambda           = self.string_to_lambda(required, 'solver')
#         self._validity_condition_lambda = self.string_to_lambda(validity_condition, 'value')

#     def string_to_lambda(self, string, var_name):
#         """
#         Converts a string representing a lambda function with a single input
#         variable into a lambda function
#         """
#         if type(string) is not str or type(var_name) is not str:
#             raise Exception(f"string and var_name must both have type str")

#         return eval('lambda ' + var_name + ': ' + string)

class SolverSystem(object):
    def __init__(self, T=None, initial_dt=None, multiphysics_solver=None, 
                 nonlinear_solver=None, linear_solver=None, 
                 ignore_surface_diffusion=False, auto_preintegrate=True):

        self.T = T
        self.initial_dt = initial_dt
        self.multiphysics_solver = multiphysics_solver
        self.nonlinear_solver = nonlinear_solver
        self.linear_solver = linear_solver

        # General settings 
        self.ignore_surface_diffusion = ignore_surface_diffusion # setting to True will treat surface variables as ODEs 

    def check_solver_system_validity(self):
        if self.initial_dt <= 0:
            raise ValueError("initial_dt must be > 0")

        for solver in [multiphysics_solver, nonlinear_solver, linear_solver]:
            if solver is not None:
                solver.check_validity()



class Solver(object):
    def __init__(self, framework='dolfin'):
        self.framework = framework
        #self.method = Setting(None, str, required='True')
    # def check_validity(self, verbose=False):
    #     """
    #     Check that all current settings are valid
    #     """
    #     for setting_name, setting in self.__dict__.items():
    #         if type(setting) == Setting:
    #             # type checks
    #             if type(setting.value) != setting.type_:
    #                 raise Exception(f"The type of setting \"{setting_name}\" is \"{type(setting.value)}\". \
    #                                   Required type is \"{setting.type_}\"")

    #             # validity condition check
    #             if setting._validity_condition_lambda(setting.value) not in (True, False):
    #                 raise Exception(f"The validity condition for setting \"{setting_name}\" must return either True or False")
    #             if setting._validity_condition_lambda(setting.value) is False:
    #                 raise Exception(f"The validity condition for setting \"{setting_name}\" was not satisfied.")

    #             # requirement/dependency check
    #             if setting._required_lambda(self) not in (True, False):
    #                 raise Exception(f"The valdiity condition for setting \"{setting_name}\" must return either True or False")
    #             if setting._required_lambda(self) is False:
    #                 print(f"Setting \"{setting_name}\" was provided but is not required.")
    #             elif setting._required_lambda(self) is True:
    #                 if setting.value is None:
    #                     raise Exception(f"Required setting \"{setting_name}\" has not been initialized!")

    #             # print a summary
    #             if verbose is True:
    #                 print(f"Setting \"{setting_name}\" passes all checks and is valid:")
    #                 print(f"\t* Type of setting, {type(setting.value)} matches required type, {setting.type_.__name__}")
    #                 print(f"\t* Validity condition, {setting.validity_condition}, is satisfied.")

    #     print(f"All settings for \"{self.__class__.__name__}\" are valid!")
    def check_solver_validity(self):
        if self.framework not in ['dolfin']:
            raise ValueError(f"Framework {self.framework} is not supported.")

        print('Solver settings are valid')

class MultiphysicsSolver(Solver):
    def __init__(self, mp_method='iterative'):
        super().__init__(mp_method)
        # default values
        self.mp_method = mp_method
        #self.method.value               = method
        #self.method.validity_condition  = "value in ['iterative', 'monolithic']"

        #self.iteration_tol = Setting(1e-6, float, required="solver.method.value=='iterative'",
        #                             validity_condition="value >= 0.0")
    def check_multiphysics_solver_validity(self):
        if method not in ['iterative', 'monolithic']:
            raise ValueError(f"MultiphysicsSolver must be either 'iterative' or 'monolithic'.")

        print('Multiphysics solver settings are valid')

class NonlinearSolver(Solver):
    def __init__(self, method='newton', min_nonlinear=2, max_nonlinear=10,
                 dt_increase_factor=1.0, dt_decrease_factor=0.8,):
        super().__init__(method)

        #self.relative_tolerance = Setting(1e-8, float, required="solver.method.value=='picard'")
        #self.absolute_tolerance = Setting(1e-8, float, required="solver.method.value=='picard'")
    def check_nls_validity(self, verbose=False):
        """
        Check that all current settings are valid
        """
        if type(self.min_nonlinear) != int or type(self.max_nonlinear) != int:
            raise TypeError("min_nonlinear and max_nonlinear must be integers.")

        if self.dt_increase_factor < 1.0: 
            raise ValueError("dt_increase_factor must be >= 1.0")

        print("All NonlinearSolver settings are valid!")
        # check settings of parent class
        self.check_solver_validity() 

class NonlinearNewtonSolver(NonlinearSolver):
    """
    Settings for dolfin nonlinear Newton solver
    """
    def __init__(self, maximum_iterations=50, error_on_nonconvergence=False, relative_tolerance=1e-8, 
                 absolute_tolerance=1e-10):
        self.method = 'newton'
        self.maximum_iterations         = maximum_iterations
        self.error_on_nonconvergence    = error_on_nonconvergence
        self.relative_tolerance         = relative_tolerance
        self.absolute_tolerance         = absolute_tolerance
    def check_validity(self):
        assert hasattr(self, 'framework')
        assert self.framework in ['dolfin']
        if type(self.maximum_iterations) != int:
            raise TypeError("maximum_iterations must be an int")
        if self.relative_tolerance <= 0: 
            raise ValueError("relative_tolerance must be greater than 0")
        if self.absolute_tolerance <= 0: 
            raise ValueError("absolute_tolerance must be greater than 0")

        print("All NonlinearNewtonSolver settings are valid!")
        # check settings of parent class
        self.check_nls_validity()

class NonlinearPicardSolver(NonlinearSolver):

    valid_norms = ['Linf', 'L2']

    def __init__(self, picard_norm = 'Linf'):
        super().__init__()

        if picard_norm not in valid_norms:
            raise ValueError(f"Invalid Picard norm ({picard_norm}) reasonable values are: {valid_norms}")
        self.picard_norm = picard_norm

# class LinearKrylovSolver(LinearSolver):
#     def __init__()
#                             'maximum_iterations': 100000,
#                         'error_on_nonconvergence': False,
#                         'nonzero_initial_guess': True,
#                         'relative_tolerance': 1e-8,
#                         'absolute_tolerance': 1e-10,

class LinearSolver(Solver):
    def __init__(self, method):
        super().__init__(Solver, method)



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


class DolfinKrylovSolver(LinearSolver):
    def __init__(self, maximum_iterations=100000, error_on_nonconvergence=False,
                 nonzero_initial_guess=True, relative_tolerance=1e-6, 
                 absolute_tolerance=1e-8):
        self.maximum_iterations         = maximum_iterations
        self.error_on_nonconvergence    = error_on_nonconvergence
        self.nonzero_initial_guess      = nonzero_initial_guess
        self.relative_tolerance         = relative_tolerance
        self. absolute_tolerance        = absolute_tolerance


    # def generate_model(self):
    #     """
    #     Processes the parameters, species, compartments, and reactions before
    #     equations are generated.
    #     """

    #     if not all([x in self.model.keys() for x in ['parameters', 'species', 'compartments', 'reactions']]):
    #         raise Exception("Parameters, species, compartments, and reactions must all be specified.")
    #     PD = self._json_to_ObjectContainer(self.model['parameters'], 'parameters')
    #     SD = self._json_to_ObjectContainer(self.model['species'], 'species')
    #     CD = self._json_to_ObjectContainer(self.model['compartments'], 'compartments')
    #     RD = self._json_to_ObjectContainer(self.model['reactions'], 'reactions')

    #     # parameter/unit assembly
    #     PD.do_to_all('assemble_units', {'unit_name': 'unit'})
    #     PD.do_to_all('assemble_units', {'value_name':'value', 'unit_name':'unit', 'assembled_name': 'value_unit'})
    #     PD.do_to_all('assembleTimeDependentParameters')
    #     SD.do_to_all('assemble_units', {'unit_name': 'concentration_units'})
    #     SD.do_to_all('assemble_units', {'unit_name': 'D_units'})
    #     CD.do_to_all('assemble_units', {'unit_name':'compartment_units'})
    #     RD.do_to_all('initialize_flux_equations_for_known_reactions', {"reaction_database": self.reaction_database})

    #     # linking containers with one another
    #     RD.link_object(PD,'param_dict','name','param_dict_values', value_is_key=True)
    #     SD.link_object(CD,'compartment_name','name','compartment')
    #     SD.copy_linked_property('compartment', 'dimensionality', 'dimensionality')
    #     RD.do_to_all('get_involved_species_and_compartments', {"SD": SD, "CD": CD})
    #     RD.link_object(SD,'involved_species','name','involved_species_link')
    #     #RD.do_to_all('combineDicts', {'dict1': 'param_dict_values', 'dict2': 'involved_species_link', 'new_dict_name': 'varDict'})

    #     # meshes
    #     CD.add_property('meshes', self.mesh)
    #     #for mesh_name in self.mesh.keys():
    #     CD.load_mesh('cyto', self.mesh['cyto'])
    #     CD.extract_submeshes('cyto', False)
    #     #CD.load_mesh('cyto', self.mesh['cyto'])
    #     #CD.extract_submeshes('cyto', False)
    #     CD.compute_scaling_factors()

    #     num_species_per_compartment = RD.get_species_compartment_counts(SD, CD, self.settings)
    #     CD.get_min_max_dim()
    #     SD.assemble_compartment_indices(RD, CD, self.settings)
    #     CD.add_property_to_all('is_in_a_reaction', False)
    #     CD.add_property_to_all('V', None)

    #     #RD.replace_sub_species_in_reactions(SD)
    #     #CD.Print()

    #     SD.assemble_dolfin_functions(RD, CD, self.settings)
    #     SD.assign_initial_conditions()

    #     RD.reaction_to_fluxes()
    #     RD.do_to_all('reaction_to_fluxes')
    #     FD = RD.get_flux_container()
    #     FD.do_to_all('get_additional_flux_properties', {"CD": CD, "config": self})

    #     # # opportunity to make custom changes

    #     FD.do_to_all('flux_to_dolfin', {"config": self})
    #     FD.check_and_replace_sub_species(SD, CD, self)

    #     model = model_assembly.Model(PD, SD, CD, RD, FD, self)

    #     # to deal with possible floating point error in mesh coordinates
    #     model.set_allow_extrapolation()
    #     # Turn fluxes into fenics/dolfin expressions
    #     model.assemble_reactive_fluxes()
    #     model.assemble_diffusive_fluxes()
    #     #model.establish_mappings()
    #     # Sort forms by type (diffusive, time derivative, etc.)
    #     model.sort_forms()

    #     model.FD.print()

    #     return model
    #     if rank==root:
    #         Print("Model created succesfully! :)")
    #         model.PD.print()
    #         model.SD.print()
    #         model.CD.print()
    #         model.RD.print()
    #     return model

    #     # ====== From here on out add to Solver class



    # def _json_to_ObjectContainer(self, json_file_name, data_type=None):
    #     if not data_type:
    #         raise Exception("Please include the type of data this is (parameters, species, compartments, reactions).")
    #     if not os.path.exists(json_file_name):
    #         raise Exception("Cannot find JSON file, %s"%json_file_name)
    #     df = read_json(json_file_name).sort_index()
    #     df = nan_to_none(df)
    #     if data_type in ['parameters', 'parameter', 'param', 'p']:
    #         return model_assembly.ParameterContainer(df)
    #     elif data_type in ['species', 'sp', 'spec', 's']:
    #         return model_assembly.SpeciesContainer(df)
    #     elif data_type in ['compartments', 'compartment', 'comp', 'c']:
    #         return model_assembly.CompartmentContainer(df)
    #     elif data_type in ['reactions', 'reaction', 'r', 'rxn']:
    #         return model_assembly.ReactionContainer(df)
    #     else:
    #         raise Exception("I don't know what kind of ObjectContainer this .json file should be")
#
## TODO: probably best to make this a class with methods e.g. (forward step)
#def picard_loop(cdf,pdf,u,t,dt,T,dT,data):
#    t += dt
#    T.assign(t)
#    dT.assign(dt)
#
#    updateTimeDependentParameters(pdf, t)
#
#    minDim = cdf.dimensionality.min()
#    pidx = 0
#    compNames = ['cyto', 'pm']
#    skipAssembly = False
#
#    while True:
#        pidx += 1
#        for compName in compNames:#u.keys():
#            comp = cdf.loc[cdf.compartment_name==compName].squeeze()
#            # solve
#            data.timer.start()
#            numIter=5 if comp.dimensionality == minDim else 1 # extra steps
#            for idx in range(numIter):
#                if skipAssembly and compName=='cyto':
#                    d.solve(A,u['cyto']['u'].vector(),b,'cg','ilu')
#                    b = d.assemble(comp.L)
#                else:
#                    d.solve(comp.a==comp.L, u[compName]['u'],
#                        solver_parameters=data.model_parameters['solver']['dolfin_params'])
#                # compute error
#                data.computeError(u,compName,data.model_parameters['solver']['norms'])
#                # assign new solution to temp variable
#                u[compName]['k'].assign(u[compName]['u'])
#                if comp.dimensionality > minDim:
#                    u[compName]['b'].interpolate(u[compName]['k'])
#
#            print('Component %s solved for in %f seconds'%(compName,data.timer.stop()))
#
#        # check convergence
#        #isConvergedRel=[] TODO: implement relative norm convergence conditions
#        isConvergedAbs=[]
#        abs_err_dict = {}
#        for compName in compNames:#u.keys():
#            for norm in data.model_parameters['solver']['norms']:
#                abs_err = data.errors[compName][norm][-1]
#                print('%s norm (%s) : %f ' % (norm, compName, abs_err))
#                isConverged = abs_err < data.model_parameters['solver']['linear_abstol']
#                abs_err_dict[compName] = abs_err
#                isConvergedAbs.append(isConverged)
#            if compName=='pm' and abs_err<1e-7 and not skipAssembly: # TODO: make this more robust
#                print('SKIPPING ASSEMBLY from now on')
#                acyto = cdf.loc[cdf.compartment_name=='cyto','a'].squeeze()
#                Lcyto = cdf.loc[cdf.compartment_name=='cyto','L'].squeeze()
#                A=d.assemble(acyto)
#                b=d.assemble(Lcyto)
#                skipAssembly = True
#
#
#        # exit picard loop if convergence criteria are met
#        if all(isConvergedAbs):
#            print('Change in norm less than tolerance (%f). Converged in %d picard iterations.'
#                % (data.model_parameters['solver']['linear_abstol'], pidx))
#            # increase time-step if convergence was quick
#            if pidx <= data.model_parameters['solver']['min_picard']:
#                dt *= data.model_parameters['solver']['dt_increase_factor']
#                print('Solution converged in less than "min_picard" iterations. Increasing time-step by a factor of %f [dt=%f]'
#                    % (data.model_parameters['solver']['dt_increase_factor'], dt))
#
#            break
#
#        # decrease time-step if not converging
#        isErrorTooHigh = [err>100 for err in list(abs_err_dict.values())]
#        if pidx >= data.model_parameters['solver']['max_picard'] or any(isErrorTooHigh):
#            # reset
#            t -= dt
#            dt *= data.model_parameters['solver']['dt_decrease_factor']
#            t += dt
#            T.assign(t); dT.assign(dt)
#            updateTimeDependentParameters(pdf, t)
#            
#            print('Maximum number of picard iterations reached. Decreasing time-step by a factor of %f [dt=%f]'
#                % (data.model_parameters['solver']['dt_decrease_factor'], dt))
#            for compName in compNames:#u.keys():
#                comp = cdf.loc[cdf.compartment_name==compName].squeeze()
#                u[compName]['k'].assign(u[compName]['n'])
#                if comp.dimensionality > minDim:
#                    u[compName]['b'].interpolate(u[compName]['n'])
#            pidx = 0
#
#    return (t,dt,T,dT)
#
#
## def updateTimeDependentParameters(pdf, t): 
##     for idx,param in pdf.iterrows():
##         if param.is_time_dependent:
##             newValue = param.symExpr.subs({'t': t}).evalf()
##             param.dolfinConstant.get().assign(newValue)
##             print('%f assigned to time-dependent parameter %s' % (newValue, param.parameter_name))
