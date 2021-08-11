import dolfin as d
from collections import defaultdict as ddict
import petsc4py.PETSc as PETSc
Print = PETSc.Sys.Print
from termcolor import colored
import numpy as np
import operator
import sympy
from scipy.integrate import solve_ivp

import stubs
import stubs.model_assembly
import stubs.common as common
from stubs import unit as ureg
color_print = common.color_print

comm = d.MPI.comm_world
rank = comm.rank
size = comm.size
root = 0



# ==============================================================================
# ==============================================================================
# Model class. Consists of parameters, species, etc. and is used for simulation
# ==============================================================================
# ==============================================================================
class Model(object):
    def __init__(self, PD, SD, CD, RD, config, solver_system, parent_mesh=None):
        self.PD = PD
        self.SD = SD
        self.CD = CD
        self.RD = RD
        self.config = config
        # Check that solver_system is valid
        solver_system.check_solver_system_validity()
        self.solver_system = solver_system
        self.solver_system.make_dolfin_parameter_dict()

        self.parent_mesh = parent_mesh
        self.params = ddict(list)

        self.idx = 0
        self.NLidx = {} # dictionary: compartment name -> # of nonlinear iterations needed
        self.success = {} # dictionary: compartment name -> success or failure to converge nonlinear solver
        self.stopping_conditions = {'F_abs': {}, 'F_rel': {}, 'udiff_abs': {}, 'udiff_rel': {}}
        self.t = 0.0
        self.dt = solver_system.initial_dt
        self.T = d.Constant(self.t)
        self.dT = d.Constant(self.dt)
        self.final_t = solver_system.final_t
        self.linear_iterations = None
        self.reset_dt = False

        self.timers = {}
        self.timings = ddict(list)

        self.Forms = stubs.model_assembly.FormContainer()
        self.a = {}
        self.L = {}
        self.F = {}
        self.nonlinear_solver = {}
        self.linear_solver = {}
        self.scipy_odes = {}

        self.data = stubs.data_manipulation.Data(self, config)


    def initialize(self):
        # parameter/unit assembly
        print("\n\n********** Model initialization (Part 1/6) **********")
        print("Assembling parameters and units...\n")
        self.PD.do_to_all('assemble_units', {'unit_name': 'unit'})
        self.PD.do_to_all('assemble_units', {'value_name':'value', 'unit_name':'unit', 'assembled_name': 'value_unit'})
        self.PD.do_to_all('assembleTimeDependentParameters')
        self.SD.do_to_all('assemble_units', {'unit_name': 'concentration_units'})
        self.SD.do_to_all('assemble_units', {'unit_name': 'D_units'})
        self.CD.do_to_all('assemble_units', {'unit_name':'compartment_units'})
        self.RD.do_to_all('initialize_flux_equations_for_known_reactions', {"reaction_database": self.config.reaction_database})

        # linking containers with one another
        print("\n\n********** Model initialization (Part 2/6) **********")
        print("Linking different containers with one another...\n")
        self.RD.link_object(self.PD,'paramDict','name','paramDictValues', value_is_key=True)
        self.SD.link_object(self.CD,'compartment_name','name','compartment')
        self.SD.copy_linked_property('compartment', 'dimensionality', 'dimensionality')
        self.RD.do_to_all('get_involved_species_and_compartments', {"SD": self.SD, "CD": self.CD})
        self.RD.link_object(self.SD,'involved_species','name','involved_species_link')

        # meshes
        print("\n\n********** Model initialization (Part 3/6) **********")
        print("Loading in mesh and computing statistics...\n")
        setattr(self.CD, 'meshes', {self.parent_mesh.name: self.parent_mesh.mesh})
        self.CD.extract_submeshes(save_to_file=False)
        self.CD.compute_scaling_factors()

        # Associate species and compartments
        print("\n\n********** Model initialization (Part 4/6) **********")
        print("Associating species with compartments...\n")
        num_species_per_compartment = self.RD.get_species_compartment_counts(self.SD, self.CD)
        self.CD.get_min_max_dim()
        self.SD.assemble_compartment_indices(self.RD, self.CD)
        self.CD.add_property_to_all('is_in_a_reaction', False)
        self.CD.add_property_to_all('V', None)

        # dolfin functions
        print("\n\n********** Model initialization (Part 5/6) **********")
        print("Creating dolfin functions and assinging initial conditions...\n")
        self.SD.assemble_dolfin_functions(self.RD, self.CD)
        self.u = self.SD.u
        self.v = self.SD.v
        self.V = self.SD.V
        self.assign_initial_conditions()

        print("\n\n********** Model initialization (Part 6/6) **********")
        print("Assembling reactive and diffusive fluxes...\n")
        self.RD.reaction_to_fluxes()
        #self.RD.do_to_all('reaction_to_fluxes')
        self.FD = self.RD.get_flux_container()
        self.FD.do_to_all('get_additional_flux_properties', {"CD": self.CD, "solver_system": self.solver_system})
        self.FD.do_to_all('flux_to_dolfin')
 
        self.set_allow_extrapolation()
        # Turn fluxes into fenics/dolfin expressions
        self.assemble_reactive_fluxes()
        self.assemble_diffusive_fluxes()
        self.sort_forms()

        self.init_solutions_and_plots()


    def initialize_refactor(self):
        # parameter/unit assembly
        print("\n\n********** Model initialization (Part 1/6) **********")
        print("Assembling parameters and units...\n")
        self.PD.do_to_all('assemble_units', {'unit_name': 'unit'})
        self.PD.do_to_all('assemble_units', {'value_name':'value', 'unit_name':'unit', 'assembled_name': 'value_unit'})
        self.PD.do_to_all('assembleTimeDependentParameters')
        self.SD.do_to_all('assemble_units', {'unit_name': 'concentration_units'})
        self.SD.do_to_all('assemble_units', {'unit_name': 'D_units'})
        self.CD.do_to_all('assemble_units', {'unit_name':'compartment_units'})
        self.RD.do_to_all('initialize_flux_equations_for_known_reactions', {"reaction_database": self.config.reaction_database})

        # linking containers with one another
        print("\n\n********** Model initialization (Part 2/6) **********")
        print("Linking different containers with one another...\n")
        self.RD.link_object(self.PD,'paramDict','name','paramDictValues', value_is_key=True)
        self.SD.link_object(self.CD,'compartment_name','name','compartment')
        self.SD.copy_linked_property('compartment', 'dimensionality', 'dimensionality')
        self.RD.do_to_all('get_involved_species_and_compartments', {"SD": self.SD, "CD": self.CD})
        self.RD.link_object(self.SD,'involved_species','name','involved_species_link')

        # meshes
        print("\n\n********** Model initialization (Part 3/6) **********")
        print("Loading in mesh and computing statistics...\n")
        self.CD.get_min_max_dim()
        #setattr(self.CD, 'meshes', {self.mesh.name: self.mesh.mesh})
        #self.CD.extract_submeshes('cyto', save_to_file=False)
        if self.parent_mesh is not None:
            self.CD.extract_submeshes_refactor(self.parent_mesh, save_to_file=False)
        else:
            raise ValueError("There is no parent mesh.")
        self.CD.compute_scaling_factors()

        # Associate species and compartments
        print("\n\n********** Model initialization (Part 4/6) **********")
        print("Associating species with compartments...\n")
        num_species_per_compartment = self.RD.get_species_compartment_counts(self.SD, self.CD)
        self.SD.assemble_compartment_indices(self.RD, self.CD)
        self.CD.add_property_to_all('is_in_a_reaction', False)
        self.CD.add_property_to_all('V', None)

        # dolfin functions
        print("\n\n********** Model initialization (Part 5/6) **********")
        print("Creating dolfin functions and assinging initial conditions...\n")
        self.SD.assemble_dolfin_functions(self.RD, self.CD)
        self.u = self.SD.u
        self.v = self.SD.v
        self.V = self.SD.V
        self.assign_initial_conditions()

        print("\n\n********** Model initialization (Part 6/6) **********")
        print("Assembling reactive and diffusive fluxes...\n")
        self.RD.reaction_to_fluxes()
        #self.RD.do_to_all('reaction_to_fluxes')
        self.FD = self.RD.get_flux_container()
        self.FD.do_to_all('get_additional_flux_properties', {"CD": self.CD, "solver_system": self.solver_system})
        self.FD.do_to_all('flux_to_dolfin')
 
        self.set_allow_extrapolation()
        # Turn fluxes into fenics/dolfin expressions
        self.assemble_reactive_fluxes()
        self.assemble_diffusive_fluxes()
        self.sort_forms()

        self.init_solutions_and_plots()

#===============================================================================
#===============================================================================
# PROBLEM SETUP
#===============================================================================
#===============================================================================
    def assemble_reactive_fluxes(self):
        """
        Creates the actual dolfin objects for each flux. Checks units for consistency
        """
        for j in self.FD.Dict.values():
            total_scaling = 1.0 # all adjustments needed to get congruent units
            sp = j.spDict[j.species_name]
            prod = j.prod
            unit_prod = j.unit_prod
            # first, check unit consistency
            if (unit_prod/j.flux_units).dimensionless:
                setattr(j, 'scale_factor', 1*ureg.dimensionless)
                pass
            else:
                if hasattr(j, 'length_scale_factor'):
                    Print("Adjusting flux for %s by the provided length scale factor." % (j.name, j.length_scale_factor))
                    length_scale_factor = getattr(j, 'length_scale_factor')
                else:
                    if len(j.involved_compartments.keys()) < 2:
                        Print("Units of flux: %s" % unit_prod)
                        Print("Desired units: %s" % j.flux_units)
                        raise Exception("Flux %s seems to be a boundary flux (or has inconsistent units) but only has one compartment, %s."
                            % (j.name, j.destination_compartment))
                    length_scale_factor = j.involved_compartments[j.source_compartment].scale_to[j.destination_compartment]

                Print(f'\nThe flux, {j.flux_name}, from compartment {j.source_compartment} to {j.destination_compartment}, has units {colored(unit_prod.to_root_units(), "red")}... the desired units for this flux are {colored(j.flux_units, "cyan")}')

                if (length_scale_factor*unit_prod/j.flux_units).dimensionless:
                    pass
                elif (1/length_scale_factor*unit_prod/j.flux_units).dimensionless:
                    length_scale_factor = 1/length_scale_factor
                else:
                    raise Exception("Inconsitent units!")

                Print('Adjusted flux with the length scale factor ' +
                      colored("%f [%s]"%(length_scale_factor.magnitude,str(length_scale_factor.units)), "cyan") + ' to match units.\n')

                prod *= length_scale_factor.magnitude
                total_scaling *= length_scale_factor.magnitude
                unit_prod *= length_scale_factor.units*1
                setattr(j, 'length_scale_factor', length_scale_factor)

            # if units are consistent in dimensionality but not magnitude, adjust values
            if j.flux_units != unit_prod:
                unit_scaling = unit_prod.to(j.flux_units).magnitude
                total_scaling *= unit_scaling
                prod *= unit_scaling
                Print(('\nThe flux, %s, has units '%j.flux_name + colored(unit_prod, "red") +
                    "...the desired units for this flux are " + colored(j.flux_units, "cyan")))
                Print('Adjusted value of flux by ' + colored("%f"%unit_scaling, "cyan") + ' to match units.\n')
                setattr(j, 'unit_scaling', unit_scaling)
            else:
                setattr(j, 'unit_scaling', 1)

            setattr(j, 'total_scaling', total_scaling)

            # adjust sign+stoich if necessary
            prod *= j.signed_stoich

            # multiply by appropriate integration measure and test function
            if j.flux_dimensionality[0] < j.flux_dimensionality[1]:
                form_key = 'B'
            else:
                form_key = 'R'
            #prod = prod*sp.v*j.int_measure
            dolfin_flux = prod*j.int_measure

            setattr(j, 'dolfin_flux', dolfin_flux)

            BRform = -prod*sp.v*j.int_measure # by convention, terms are all defined as if they were on the LHS of the equation e.g. F(u;v)=0
            self.Forms.add(stubs.model_assembly.Form(BRform, sp, form_key, flux_name=j.name))


    def assemble_diffusive_fluxes(self):
        min_dim = min(self.CD.get_property('dimensionality').values())
        max_dim = max(self.CD.get_property('dimensionality').values())
        dT = self.dT

        for sp_name, sp in self.SD.Dict.items():
            if sp.is_in_a_reaction:
                if self.solver_system.nonlinear_solver.method in ['picard', 'IMEX']:
                    u = sp.u['t']
                elif self.solver_system.nonlinear_solver.method == 'newton':
                    u = sp.u['u']
                un = sp.u['n']
                v = sp.v
                D = sp.D

                if sp.dimensionality == max_dim:
                    dx = sp.compartment.dx
                    Dform = D*d.inner(d.grad(u), d.grad(v)) * dx
                    self.Forms.add(stubs.model_assembly.Form(Dform, sp, 'D'))
                elif sp.dimensionality < max_dim:
                    if self.solver_system.ignore_surface_diffusion:
                        dx=sp.compartment.dP
                    else:
                        dx = sp.compartment.dx
                        Dform = D*d.inner(d.grad(u), d.grad(v)) * dx
                        self.Forms.add(stubs.model_assembly.Form(Dform, sp, 'D'))

                # time derivative
                Mform_u = u/dT * v * dx
                Mform_un = -un/dT * v * dx
                self.Forms.add(stubs.model_assembly.Form(Mform_u, sp, "Mu"))
                self.Forms.add(stubs.model_assembly.Form(Mform_un, sp, "Mun"))

            else:
                Print("Species %s is not in a reaction?" %  sp_name)

    def set_allow_extrapolation(self):
        for comp_name in self.u.keys():
            ucomp = self.u[comp_name]
            for func_key in ucomp.keys():
                if func_key != 't': # trial function by convention
                    self.u[comp_name][func_key].set_allow_extrapolation(True)

    def sort_forms(self):
        """
        Organizes forms based on solution method. E.g. for picard iterations we
        split the forms into a bilinear and linear component, for Newton we
        simply solve F(u;v)=0.
        """
        comp_list = [self.CD.Dict[key] for key in self.u.keys()]
        self.split_forms = ddict(dict)
        form_types = set([f.form_type for f in self.Forms.form_list])

        if self.solver_system.nonlinear_solver.method == 'picard':
            raise Exception("Picard functionality needs to be reviewed")
            # Print("Splitting problem into bilinear and linear forms for picard iterations: a(u,v) == L(v)")
            # for comp in comp_list:
            #     comp_forms = [f.dolfin_form for f in self.Forms.select_by('compartment_name', comp.name)]
            #     self.a[comp.name] = d.lhs(sum(comp_forms))
            #     self.L[comp.name] = d.rhs(sum(comp_forms))
            #     problem = d.LinearVariationalProblem(self.a[comp.name],
            #                                          self.L[comp.name], self.u[comp.name]['u'], [])
            #     self.linear_solver[comp.name] = d.LinearVariationalSolver(problem)
            #     p = self.linear_solver[comp.name].parameters
            #     p['linear_solver'] = self.solver_system.linear_solver.method
            #     if type(self.solver_system.linear_solver) == stubs.solvers.DolfinKrylovSolver:
            #         p['krylov_solver'].update(self.solver_system.linear_solver.__dict__)
            #         p['krylov_solver'].update({'nonzero_initial_guess': True}) # important for time dependent problems

        elif self.solver_system.nonlinear_solver.method == 'newton':
            Print("Formulating problem as F(u;v) == 0 for newton iterations")
            for comp in comp_list:
                comp_forms = [f.dolfin_form for f in self.Forms.select_by('compartment_name', comp.name)]
                self.F[comp.name] = sum(comp_forms)
                J = d.derivative(self.F[comp.name], self.u[comp.name]['u'])

                problem = d.NonlinearVariationalProblem(self.F[comp.name], self.u[comp.name]['u'], [], J)

                self.nonlinear_solver[comp.name] = d.NonlinearVariationalSolver(problem)
                p = self.nonlinear_solver[comp.name].parameters
                p['nonlinear_solver'] = 'newton'
                p['newton_solver'].update(self.solver_system.nonlinear_dolfin_solver_settings)
                p['newton_solver']['krylov_solver'].update(self.solver_system.linear_dolfin_solver_settings)
                p['newton_solver']['krylov_solver'].update({'nonzero_initial_guess': True}) # important for time dependent problems

        elif self.solver_system.nonlinear_solver.method == 'IMEX':
            raise Exception("IMEX functionality needs to be reviewed")
#            Print("Keeping forms separated by compartment and form_type for IMEX scheme.")
#            for comp in comp_list:
#                comp_forms = self.Forms.select_by('compartment_name', comp.name)
#                for form_type in form_types:
#                    self.split_forms[comp.name][form_type] = sum([f.dolfin_form for f in comp_forms if f.form_type==form_type])

#===============================================================================
#===============================================================================
# SOLVING
#===============================================================================
#===============================================================================


    def solve_single_timestep(self, plot_period=1):
        # Solve using specified multiphysics scheme 
        if self.solver_system.multiphysics_solver.method == "iterative":
            self.iterative_mpsolve()
        else:
            raise Exception("I don't know what operator splitting scheme to use")

        # post processing
        self.post_process()
        if (self.idx % plot_period == 0 or self.t >= self.final_t) and plot_period!=0:
            self.plot_solution()

        # if we've reached final time
        end_simulation = True if self.t >= self.final_t else False

        return end_simulation

    def solve(self, plot_period=1):
        # Initialize
        #self.init_solutions_and_plots()

        self.stopwatch("Total simulation")
        # Time loop
        while True:
            end_simulation = self.solve_single_timestep(plot_period)

            if end_simulation:
                break

        self.stopwatch("Total simulation", stop=True)
        Print("Solver finished with %d total time steps." % self.idx)

## Yuan's code
    # def solve_2(self, plot_period=1, store_solutions=True, check_mass=False, species_to_check=None, x_compartment=None):
    #     ## solve
    #     self.init_solver_and_plots()

    #     self.stopwatch("Total simulation")
    #     self.mass=[]
    #     while True:
    #         if check_mass:
    #             assert x_compartment is not None
    #             self.mass.append((self.t, self.compute_mass_step(species_to_check=species_to_check,x_compartment=x_compartment)))
    #         #Solve using specified multiphysics scheme 
    #         if self.solver_system.multiphysics_solver.method == "iterative":
    #             self.iterative_mpsolve()
    #         else:
    #             raise Exception("I don't know what operator splitting scheme to use")


    # def solve(self, plot_period=1, store_solutions=True, check_mass=False, species_to_check=None, x_compartment=None):
    #     ## solve
    #     self.init_solver_and_plots()

    #     self.stopwatch("Total simulation")
    #     self.mass=[]
    #     while True:
    #         if check_mass:
    #             assert x_compartment is not None
    #             self.mass.append((self.t, self.compute_mass_step(species_to_check=species_to_check,x_compartment=x_compartment)))
    #         #Solve using specified multiphysics scheme 
    #         if self.solver_system.multiphysics_solver.method == "iterative":
    #             self.iterative_mpsolve()
    #         else:
    #             raise Exception("I don't know what operator splitting scheme to use")


    #         # post processing
    #         self.compute_statistics()
    #         if self.idx % plot_period == 0 or self.t >= self.final_t:
    #             self.plot_solution(store_solutions=store_solutions)
    #             self.plot_solver_status()

            

    #         # if we've reached final time
    #         if self.t >= self.final_t:
    #             break
    #         #self.solve_step(plot_period, store_solutions)
    #     if check_mass:
    #         mass_list = [i[1] for i in self.mass]
    #         assert abs((max(mass_list)-min(mass_list))/max(mass_list)) <=0.05
    #     self.stopwatch("Total simulation", stop=True)
    #     Print("Solver finished with %d total time steps." % self.idx)

    # def compute_mass_step(self,species_to_check=None, x_compartment='cyto'):
    #     #seperate function with keyword
        
        

    #     com = self.CD.Dict[x_compartment]
    #     #com_s = self.CD.Dict[s_compartment]
    #     dx = com.dx
    #     ds = com.ds

    #     if species_to_check is None:
    #         print('Assuming mass of all species are conserved, and the stiochiometry coefficient for every species is one')
    #         species_to_check = {i: 1 for i in self.SD.Dict}
    #     target_unit_dict={}
    #     coefficient = {}
    #     max_dim=com.dimensionality
    #     mass=0
    #     for i in species_to_check:
    #         s=self.SD.Dict[i]
    #         target_unit_dict[i] = ureg.molecule/com.compartment_units.units**s.dimensionality
    #         coefficient[i] = s.concentration_units.to(target_unit_dict[i]).to_tuple()[0]
    #         if s.dimensionality == max_dim:
    #             mass+=species_to_check[i]*coefficient[i]*d.assemble(s.u['u']*dx)
    #         else:
    #             mass+=species_to_check[i]*coefficient[i]*d.assemble(s.u['u']*ds(self.CD.Dict[s.compartment_name].cell_marker))
    #     ##compartment unit^comp_dim
    #     return mass


    # def solve_zero_d(self,t_span,initial_guess_for_root=None):
    #         func_vector = self.get_lambdified()[0]
    #         func_vector_t = lambda t,y:func_vector(y)

    #         return self.de_solver(func_vector_t, t_span,initial_guess_for_root)

    # def de_solver(self, func_vectors, t_span,initial_guess_for_root=None, root_check=True, max_step=0.01, jac=False, method='RK45'):
    #     #assert initial_guess_for_root is not None
    #     if initial_guess_for_root is None:
    #         print("Using the initial condition from config files. Unites will be automatically converted!")
    #     coefficient_dict={}
    #     target_unit = self.SD.Dict[list(self.SD.Dict.keys())[0]].concentration_units
    #     for i in self.SD.Dict:
    #         try:
    #             coefficient_dict[i] = self.SD.Dict[i].concentration_units.to(target_unit).to_tuple()[0]
    #         except:
    #             raise RuntimeError('Units mismatched for a well-mixed system')
    #     initial_guess_for_root = [(coefficient_dict[i] * self.SD.Dict[i].initial_condition) for i in self.SD.Dict]
    #     # if initial_guess_for_root is None:
    #     #     initial_guess_for_root = [0]*num_params
    #     #     ("Warning: The initial condition of species is not given")
    #     # if root_check:
    #     #     root_info = optimize.root(lambda y:func_vectors(0, y),initial_guess_for_root,jac=jac)
    #     #     if root_info.success:
    #     #         root = root_info.x
    #     #         if (((initial_guess_for_root - root)/initial_guess_for_root)>0.1).any():
    #     #             print("Initial guess doesn't match root condition")
    #     #     else:
    #     #         raise Exception('Unable to find initial condition: unable to find root')
    #     # else:
    #     root = initial_guess_for_root
    #     sol = solve_ivp(func_vectors, t_span, root, max_step=max_step, method=method)
    #     #returns u
    #     return sol
    
    # def get_lambdified(self):
        
    #     sps = [i for i in self.SD.Dict]
    #     func_dict = {i: None for i in list(self.SD.Dict.keys())}
    #     for f in self.FD.Dict:
    #         sp = self.FD.Dict[f].species_name
    #         if func_dict[sp] is None:
    #             func_dict[sp] = self.FD.Dict[f].symEqn*self.FD.Dict[f].signed_stoich
    #         else:
    #             func_dict[sp] += self.FD.Dict[f].symEqn*self.FD.Dict[f].signed_stoich
            
            
    #     func_vector=[]
    #     for j in func_dict:    
    #         for i in self.PD.Dict:
    #             func_dict[j] = func_dict[j].subs(i, self.PD.Dict[i].value)
            
    #         lam = sympy.lambdify(sps, func_dict[j])
    #         func_vector.append(lam)
    #     return lambda u:[f(*u) for f in func_vector], func_dict

    def set_time(self, t, dt=None):
        if not dt:
            dt = self.dt
        else:
            Print("dt changed from %f to %f" % (self.dt, dt))
        if t != self.t:
            Print("Time changed from %f to %f" % (self.t, t))
        self.t = t
        self.T.assign(t)
        self.dt = dt
        self.dT.assign(dt)

    def check_dt_resets(self):
        """
        Checks to see if the size of a full-time step would pass a "reset dt"
        checkpoint. At these checkpoints dt is reset to some value
        (e.g. to force smaller sampling during fast events)
        """
        # check that there are reset times specified
        if self.solver_system.adjust_dt is None or len(self.solver_system.adjust_dt) == 0:
            return

        # if last time-step we passed a reset dt checkpoint then reset it now
        if self.reset_dt:
            new_dt = self.solver_system.adjust_dt[0][1]
            color_print("(!!!) Adjusting time-step (dt = %f -> %f) to match config specified value" % (self.dt, new_dt), 'green')
            self.set_time(self.t, dt = new_dt)
            del(self.solver_system.adjust_dt[0])
            self.reset_dt = False
            return

        # check if we pass a reset dt checkpoint
        t0 = self.t # original time
        potential_t = t0 + self.dt # the final time if dt is not reset
        next_reset_time = self.solver_system.adjust_dt[0][0] # the next time value to reset dt at
        next_reset_dt = self.solver_system.adjust_dt[0][1] # the next value of dt to reset to
        if next_reset_time<t0:
            raise Exception("Next reset time is less than time at beginning of time-step.")
        if t0 < next_reset_time <= potential_t: # check if the full time-step would pass a reset dt checkpoint
            new_dt = max([next_reset_time - t0, next_reset_dt]) # this is needed otherwise very small time-steps might be taken which wont converge
            color_print("(!!!) Adjusting time-step (dt = %f -> %f) to avoid passing reset dt checkpoint" % (self.dt, new_dt), 'blue')
            self.set_time(self.t, dt=new_dt)
            # set a flag to change dt to the config specified value
            self.reset_dt = True

    def check_dt_pass_tfinal(self):
        """
        Check that t+dt is not > t_final
        """
        potential_t = self.t + self.dt
        if potential_t > self.final_t:
            new_dt = self.final_t - self.t 
            color_print("(!!!) Adjusting time-step (dt = %f -> %f) to avoid passing final time" % (self.dt, new_dt), 'blue')
            self.set_time(self.t, dt=new_dt)

    def forward_time_step(self, dt_factor=1):

        self.dT.assign(float(self.dt*dt_factor))
        self.t = float(self.t+self.dt*dt_factor)
        self.T.assign(self.t)
        self.updateTimeDependentParameters(dt=self.dt*dt_factor)
        Print("t: %f , dt: %f" % (self.t, self.dt*dt_factor))

    def stopwatch(self, key, stop=False, color='cyan'):
        if key not in self.timers.keys():
            self.timers[key] = d.Timer()
        if not stop:
            self.timers[key].start()
        else:
            elapsed_time = self.timers[key].elapsed()[0]
            time_str = str(elapsed_time)[0:8]
            if color:
                time_str = colored(time_str, color)
            Print("%s finished in %s seconds" % (key,time_str))
            self.timers[key].stop()
            self.timings[key].append(elapsed_time)
            return elapsed_time

    def updateTimeDependentParameters(self, t=None, t0=None, dt=None):
        """ Updates all time dependent parameters. Time-dependent parameters are 
        either defined either symbolically or through a data file, and each of 
        these can either be defined as a direct function of t, p(t), or a 
        "pre-integrated expression", \int_{0}^{t} P(x) dx, which allows for 
        exact integration when the expression the parameter appears in is purely
        time-dependent.
        
        Args:
            t (float, optional): Current time in the simulation to update parameters to 
            t0 (float, optional): Previous time value (i.e. t0 == t-dt)
            dt (float, optional): Time step
        
        Raises:
            Exception: Description
        """
        if t is None:
            t = self.t
        if t0 is not None and dt is not None:
            raise Exception("Specify either t0 or dt, not both.")
        elif t0 is not None:
            dt = t-t0
        elif dt is not None:
            t0 = t-dt
        if t0 is not None:
            if t0<0 or dt<=0:
                raise Exception("Either t0<0 or dt<=0, is this the desired behavior?")

        # Update time dependent parameters
        for param_name, param in self.PD.Dict.items():
            # check to make sure a parameter wasn't assigned a new value more than once
            value_assigned = 0
            if not param.is_time_dependent:
                continue
            # use value by evaluating symbolic expression
            if param.symExpr and not param.preintegrated_symExpr:
                newValue = float(param.symExpr.subs({'t': t}).evalf())
                value_assigned += 1
                print(f"Parameter {param_name} assigned by symbolic expression")

            # calculate a preintegrated expression by subtracting previous value
            # and dividing by time-step
            if param.symExpr and param.preintegrated_symExpr:
                if t0 is None:
                    raise Exception("Must provide a time interval for"\
                                    "pre-integrated variables.")
                a = param.preintegrated_symExpr.subs({'t': t0}).evalf()
                b = param.preintegrated_symExpr.subs({'t': t}).evalf()
                newValue = float((b-a)/dt)
                value_assigned += 1
                print(f"Parameter {param_name} assigned by preintegrated symbolic expression")

            # if parameter is given by data
            if param.sampling_data is not None and param.preint_sampling_data is None:
                data = param.sampling_data
                # We never want time to extrapolate beyond the provided data.
                if t<data[0,0] or t>data[-1,0]:
                    raise Exception("Parameter cannot be extrapolated beyond"\
                                    "provided data.")
                # Just in case... return a nan if value is outside of bounds
                newValue = float(np.interp(t, data[:,0], data[:,1],
                                     left=np.nan, right=np.nan))
                value_assigned += 1
                print(f"Parameter {param_name} assigned by sampling data")

            # if parameter is given by data and it has been pre-integrated
            if param.sampling_data is not None and param.preint_sampling_data is not None:
                int_data = param.preint_sampling_data
                a = np.interp(t0, int_data[:,0], int_data[:,1],
                                     left=np.nan, right=np.nan)
                b = np.interp(t, int_data[:,0], int_data[:,1],
                                     left=np.nan, right=np.nan)
                newValue = float((b-a)/dt)
                value_assigned += 1
                print(f"Parameter {param_name} assigned by preintegrated sampling data")

            if value_assigned != 1:
                raise Exception("Either a value was not assigned or more than"\
                                "one value was assigned to parameter %s" % param.name)

            Print('%f assigned to time-dependent parameter %s'
                  % (newValue, param.name))

            if np.isnan(newValue):
                raise ValueError(f"Warning! Parameter {param_name} is a NaN.")

            param.value = newValue
            param.dolfinConstant.assign(newValue)
            self.params[param_name].append((t,newValue))


    def reset_timestep(self, comp_list=[]):
        """
        Resets the time back to what it was before the time-step. Optionally, input a list of compartments
        to have their function values reset (['n'] value will be assigned to ['u'] function).
        """
        self.set_time(self.t - self.dt, self.dt*self.solver_system.nonlinear_solver.dt_decrease_dt_factor)
        Print("Resetting time-step and decreasing step size")
        for comp_name in comp_list:
            self.u[comp_name]['u'].assign(self.u[comp_name]['n'])
            Print("Assigning old value of u to species in compartment %s" % comp_name)

    def update_solution_boundary_to_volume(self):
        for comp_name in self.CD.Dict.keys():
            for key in self.u[comp_name].keys():
                if key[0:2] == 'v_': # fixme
                    d.LagrangeInterpolator.interpolate(self.u[comp_name][key], self.u[comp_name]['u'])
                    parent_comp_name = key[2:]
                    Print("Projected values from surface %s to volume %s" % (comp_name, parent_comp_name))

    def update_solution_volume_to_boundary(self):
        for comp_name in self.CD.Dict.keys():
            for key in self.u[comp_name].keys():
                if key[0:2] == 'b_': # fixme
                    #self.u[comp_name][key].interpolate(self.u[comp_name]['u'])
                    d.LagrangeInterpolator.interpolate(self.u[comp_name][key], self.u[comp_name]['u'])
                    sub_comp_name = key[2:]
                    Print("Projected values from volume %s to surface %s" % (comp_name, sub_comp_name))

    def boundary_reactions_forward(self, dt_factor=1, bcs=[]):
        self.stopwatch("Boundary reactions forward")

        # solve boundary problem(s)
        for comp_name, comp in self.CD.Dict.items():
            if comp.dimensionality < self.CD.max_dim:
                self.nonlinear_solve(comp_name, dt_factor=dt_factor)
        self.stopwatch("Boundary reactions forward", stop=True)

    def volume_reactions_forward(self, dt_factor=1, bcs=[], time_step=True):
        self.stopwatch("Volume reactions forward")

        # solve volume problem(s)
        for comp_name, comp in self.CD.Dict.items():
            if comp.dimensionality == self.CD.max_dim:
                self.nonlinear_solve(comp_name, dt_factor=dt_factor)
        self.stopwatch("Volume reactions forward", stop=True)

    def nonlinear_solve(self, comp_name, dt_factor=1.0):
        """
        A switch for choosing a nonlinear solver
        """

        if self.solver_system.nonlinear_solver.method == 'newton':
            self.stopwatch("Newton's method [%s]" % comp_name)
            self.NLidx[comp_name], self.success[comp_name] = self.nonlinear_solver[comp_name].solve()
            self.stopwatch("Newton's method [%s]" % comp_name, stop=True)
            Print(f"{self.NLidx[comp_name]} Newton iterations required for convergence on compartment {comp_name}.")
        # elif self.solver_system.nonlinear_solver.method == 'picard':
        #     self.stopwatch("Picard iteration method [%s]" % comp_name)
        #     self.NLidx[comp_name], self.success[comp_name] = self.picard_loop(comp_name, dt_factor=dt_factor)
        #     self.stopwatch("Picard iteration method [%s]" % comp_name)
        #     Print(f"{self.NLidx[comp_name]} Newton iterations required for convergence on compartment {comp_name}.")
        else:
            raise ValueError("Unknown nonlinear solver method")

    def update_solution(self, ukeys=['k', 'n']):
        """
        After finishing a time step, assign all the most recently computed solutions as 
        the solutions for the previous time step.
        """
        for key in self.u.keys():
            for ukey in ukeys:
                self.u[key][ukey].assign(self.u[key]['u'])

    def adjust_dt(self):
        ## check if time step should be changed
        #if all([x <= self.solver_system.nonlinear_solver.min_nonlinear for x in self.NLidx.values()]):
        #    self.set_time(self.t, dt=self.dt*self.solver_system.nonlinear_solver.dt_increase_factor)
        #    Print("Increasing step size")
        #elif any([x > self.solver_system.nonlinear_solver.max_nonlinear for x in self.NLidx.values()]):
        #    self.set_time(self.t, dt=self.dt*self.solver_system.nonlinear_solver.dt_decrease_factor)
        #    Print("Decreasing step size")

        # check if time step should be changed based on number of multiphysics iterations
        if self.mpidx <= self.solver_system.multiphysics_solver.min_multiphysics:
            self.set_time(self.t, dt=self.dt*self.solver_system.multiphysics_solver.dt_increase_factor)
            Print("Increasing step size")
        elif self.mpidx > self.solver_system.multiphysics_solver.max_multiphysics:
            self.set_time(self.t, dt=self.dt*self.solver_system.multiphysics_solver.dt_decrease_factor)
            Print("Decreasing step size")

    def compute_stopping_conditions(self, norm_type=np.inf):

        for comp_name in self.u.keys():
            # dolfin norm() is not really robust so we get vectors and use numpy.linalg.norm
            #Fabs = d.norm(d.assemble(self.F[comp_name]), norm_type)
            #udiff = self.u[comp_name]['u'] - self.u[comp_name]['k']
            #udiff_abs = d.norm(udiff, norm_type)
            #udiff_rel = udiff_abs/d.norm(self.u[comp_name]['u'], norm_type)

            Fvec = d.assemble(self.F[comp_name]).get_local()

            Fabs = np.linalg.norm(Fvec, ord=norm_type)

            self.stopping_conditions['F_abs'].update({comp_name: Fabs})
            #color_print(f"{'Computed F_abs for component '+comp_name+': ': <40} {Fabs:.4e}", color='green')

        for sp_name, sp in self.SD.Dict.items():
            uvec = self.dolfin_get_function_values(sp, ukey='u')
            ukvec = self.dolfin_get_function_values(sp, ukey='k')
            udiff = uvec - ukvec

            udiff_abs = np.linalg.norm(udiff, ord=norm_type)
            udiff_rel = udiff_abs/np.linalg.norm(uvec, ord=norm_type)

            self.stopping_conditions['udiff_abs'].update({sp_name: udiff_abs})
            self.stopping_conditions['udiff_rel'].update({sp_name: udiff_rel})
            #color_print(f"{'Computed udiff_abs for species '+comp_name+': ': <40} {udiff_abs:.4e}", color='green')
            #color_print(f"{'Computed udiff_rel for species '+comp_name+': ': <40} {udiff_rel:.4e}", color='green')

    def iterative_mpsolve(self, bcs=[]):
        """
        Iterate between boundary and volume problems until convergence
        """
        Print('\n\n\n')
        self.idx += 1
        self.check_dt_resets()      # check if there is a manually prescribed time-step size
        self.check_dt_pass_tfinal() # check that we don't pass tfinal
        color_print(f'\n *** Beginning time-step {self.idx} [time={self.t}, dt={self.dt}] ***\n', color='red')

        self.stopwatch("Total time step") # start a timer for the total time step

        self.forward_time_step() # march forward in time and update time-dependent parameters

        self.mpidx = 0
        while True: 
            self.mpidx += 1
            color_print(f'\n * Multiphysics iteration {self.mpidx} for time-step {self.idx} [time={self.t}] *', color='green')
            # solve volume problem(s)
            self.volume_reactions_forward()
            self.update_solution_volume_to_boundary()
            # solve boundary problem(s)
            self.boundary_reactions_forward()
            self.update_solution_boundary_to_volume()
            # decide whether to stop iterations
            self.compute_stopping_conditions()
            if self.solver_system.multiphysics_solver.eps_Fabs is not None:
                if all([x<self.solver_system.multiphysics_solver.eps_Fabs for x in self.stopping_conditions['F_abs'].values()]):
                    color_print(f"All F_abs are below tolerance, {self.solver_system.multiphysics_solver.eps_Fabs}." \
                                  f" Exiting multiphysics loop ({self.mpidx} iterations).\n", color='green')
                    break
                else: 
                    max_comp_name, max_Fabs = max(self.stopping_conditions['F_abs'].items(), key=operator.itemgetter(1))
                    color_print(f"{'One or more F_abs are above tolerance. Max F_abs is from component '+max_comp_name+': ': <40} {max_Fabs:.4e}", color='green')

            if self.solver_system.multiphysics_solver.eps_udiff_abs is not None:
                if all([x<self.solver_system.multiphysics_solver.eps_udiff_abs for x in self.stopping_conditions['udiff_abs'].values()]):
                    color_print(f"All udiff_abs are below tolerance, {self.solver_system.multiphysics_solver.eps_udiff_abs}." \
                                  f" Exiting multiphysics loop ({self.mpidx} iterations).\n", color='green')
                    break
                else: 
                    max_sp_name, max_udiffabs = max(self.stopping_conditions['udiff_abs'].items(), key=operator.itemgetter(1))
                    color_print(f"{'One or more udiff_abs are above tolerance. Max udiff_abs is from species '+max_sp_name+': ': <40} {max_udiffabs:.4e}", color='green')

            if self.solver_system.multiphysics_solver.eps_udiff_rel is not None:
                if all([x<self.solver_system.multiphysics_solver.eps_udiff_rel for x in self.stopping_conditions['udiff_rel'].values()]):
                    color_print(f"All udiff_rel are below tolerance, {self.solver_system.multiphysics_solver.eps_udiff_rel}." \
                                  f" Exiting multiphysics loop ({self.mpidx} iterations).\n", color='green')
                    break
                else: 
                    max_sp_name, max_udiffrel = max(self.stopping_conditions['udiff_rel'].items(), key=operator.itemgetter(1))
                    color_print(f"{'One or more udiff_rel are above tolerance. Max udiff_rel is from species '+max_sp_name+': ': <40} {max_udiffrel:.4e}", color='green')

            # update previous (iteration) solution 
            self.update_solution(ukeys=['k'])


        self.update_solution() # assign most recently computed solution to "previous solution"
        self.adjust_dt() # adjusts dt based on number of nonlinear iterations required
        self.stopwatch("Total time step", stop=True, color='cyan')


#===============================================================================
#===============================================================================
# Nonlinear solvers:
#  - Timestep
#  - Update time dependent parameters
#  - Solve
#  - Assign new solution to old solution unless otherwise stated
#===============================================================================
#===============================================================================



#     def picard_loop(self, comp_name, dt_factor=1, bcs=[]):
#         """
#         Continue picard iterations until a specified tolerance or count is
#         reached.
#         """
#         self.stopwatch("Picard loop [%s]" % comp_name)
#         t0 = self.t
#         self.forward_time_step(dt_factor=dt_factor) # increment time afterwards
#         self.pidx = 0 # count the number of picard iterations
#         success = True

#         # main loop
#         while True:
#             self.pidx += 1
#             #linear_solver_settings = self.config.dolfin_krylov_solver

#             # solve
#             self.linear_solver[comp_name].solve()
#             #d.solve(self.a[comp_name]==self.L[comp_name], self.u[comp_name]['u'], bcs, solver_parameters=linear_solver_settings)

#             # update temporary value of u
#             self.data.computeError(self.u, comp_name, self.solver_system.nonlinear_solver.picard_norm)
#             self.u[comp_name]['k'].assign(self.u[comp_name]['u'])

#             # Exit if error tolerance or max iterations is reached
#             Print('Linf norm (%s) : %f ' % (comp_name, self.data.errors[comp_name]['Linf']['abs'][-1]))
#             if self.data.errors[comp_name]['Linf']['abs'][-1] < self.solver_system.nonlinear_solver.absolute_tolerance:
#                 #print("Norm (%f) is less than linear_abstol (%f), exiting picard loop." %
#                  #(self.data.errors[comp_name]['Linf'][-1], self.config.solver['linear_abstol']))
#                 break
# #            if self.data.errors[comp_name]['Linf']['rel'][-1] < self.config.solver['linear_reltol']:
# #                print("Norm (%f) is less than linear_reltol (%f), exiting picard loop." %
# #                (self.data.errors[comp_name]['Linf']['rel'][-1], self.config.solver['linear_reltol']))
# #                break

#             if self.pidx >= self.solver_system.nonlinear_solver.maximum_iterations:
#                 Print("Max number of picard iterations reached (%s), exiting picard loop with abs error %f." %
#                 (comp_name, self.data.errors[comp_name]['Linf']['abs'][-1]))
#                 success = False
#                 break

#         self.stopwatch("Picard loop [%s]" % comp_name, stop=True)

#         Print("%d Picard iterations required for convergence." % self.pidx)
#         return self.pidx, success


#===============================================================================
#===============================================================================
# POST-PROCESSING
#===============================================================================
#===============================================================================
    def init_solutions_and_plots(self):
        self.data.initSolutionFiles(self.SD, self.config)
        self.data.storeSolutionFiles(self.u, self.t, self.config)
        self.data.compute_statistics(self.u, self.t, self.dt, self.SD, self.PD, self.CD, self.FD, self.NLidx)
        self.data.initPlot(self.config, self.SD, self.FD)

    def post_process(self):
        self.data.compute_statistics(self.u, self.t, self.dt, self.SD, self.PD, self.CD, self.FD, self.NLidx)
        self.data.compute_probe_values(self.u, self.SD)
        self.data.outputPickle()
        self.data.outputCSV()

    def plot_solution(self):
        self.data.storeSolutionFiles(self.u, self.t, self.config)
        self.data.plotParameters(self.config)
        self.data.plotSolutions(self.config, self.SD)
        self.data.plotFluxes(self.config)
        self.data.plotSolverStatus(self.config)




##### DATA_MANIPULATION

    # get the values of function u from subspace idx of some mixed function space, V
    def dolfin_get_dof_indices(self, sp):#V, species_idx):
        """
        Returned indices are *local* to the CPU (not global)
        function values can be returned e.g.
        indices = dolfin_get_dof_indices(V,species_idx)
        u.vector().get_local()[indices]
        """
        V           = sp.compartment.V
        species_idx = sp.compartment_index

        if V.num_sub_spaces() > 1:
            indices = np.array(V.sub(species_idx).dofmap().dofs())
        else:
            indices = np.array(V.dofmap().dofs())
        first_idx, last_idx = V.dofmap().ownership_range() # indices that this CPU owns

        return indices-first_idx # subtract index offset to go from global -> local indices


    def reduce_vector(u):
        """
        comm.allreduce() only works when the incoming vectors all have the same length. We use comm.Gatherv() to gather vectors
        with different lengths
        """
        sendcounts = np.array(comm.gather(len(u), root)) # length of vectors being sent by workers
        if rank == root:
            print("reduceVector(): CPUs sent me %s length vectors, total length: %d"%(sendcounts, sum(sendcounts)))
            recvbuf = np.empty(sum(sendcounts), dtype=float)
        else:
            recvbuf = None

        comm.Gatherv(sendbuf=u, recvbuf=(recvbuf, sendcounts), root=root)

        return recvbuf

    def dolfin_get_function_values(self, sp, ukey='u'):
        """
        Returns the values from a sub-function of a VectorFunction. When run
        in parallel this will *not* double-count overlapping vertices which
        are shared by multiple CPUs (a simple call to 
        u.sub(species_idx).compute_vertex_values() will double-count...)
        """
        # Get the VectorFunction
        V           = sp.compartment.V
        species_idx = sp.compartment_index
        u           = self.u[sp.compartment_name][ukey]
        indices     = self.dolfin_get_dof_indices(sp)
        uvec        = u.vector().get_local()[indices]
        return uvec

    def dolfin_get_function_values_at_point(self, sp, coord):
        """
        Returns the values of a dolfin function at the specified coordinate 
        :param dolfin.function.function.Function u: Function to extract values from
        :param tuple coord: tuple of floats indicating where in space to evaluate u e.g. (x,y,z)
        :param int species_idx: index of species
        :return: list of values at point. If species_idx is not specified it will return all values
        """
        u = self.u[sp.compartment_name]['u']
        if sp.compartment.V.num_sub_spaces() == 0:
            return u(coord)
        else:
            species_idx = sp.compartment_index
            return u(coord)[species_idx]

    def dolfin_set_function_values(self, sp, ukey, unew):
        """
        unew can either be a scalar or a vector with the same length as u
        """

        u = self.u[sp.compartment_name][ukey]

        indices = self.dolfin_get_dof_indices(sp)
        uvec    = u.vector()
        values  = uvec.get_local()
        values[indices] = unew

        uvec.set_local(values)
        uvec.apply('insert')

    def assign_initial_conditions(self):
        ukeys = ['k', 'n', 'u']
        for sp_name, sp in self.SD.Dict.items():
            comp_name = sp.compartment_name
            for ukey in ukeys:
                self.dolfin_set_function_values(sp, ukey, sp.initial_condition)
            if rank==root: print("Assigned initial condition for species %s" % sp.name)

        # project to boundary/volume functions
        self.update_solution_volume_to_boundary()
        self.update_solution_boundary_to_volume()

    # def dolfinFindClosestPoint(mesh, coords):
    #     """
    #     Given some point and a mesh, returns the coordinates of the nearest vertex
    #     """
    #     p = d.Point(coords)
    #     L = list(d.vertices(mesh))
    #     distToVerts = [np.linalg.norm(p.array() - x.midpoint().array()) for x in L]
    #     minDist = min(distToVerts)
    #     minIdx = distToVerts.index(minDist) # returns the local index (wrt the cell) of the closest point

    #     closestPoint = L[minIdx].midpoint().array()

    #     if size > 1:
    #         min_dist_global, min_idx = comm.allreduce((minDist,rank), op=pyMPI.MINLOC)

    #         if rank == min_idx:
    #             comm.Send(closestPoint, dest=root)

    #         if rank == root:
    #             comm.Recv(closestPoint, min_idx)
    #             print("CPU with rank %d has the closest point to %s: %s. The distance is %s" % (min_idx, coords, closestPoint, min_dist_global))
    #             return closestPoint, min_dist_global
    #     else:
    #         return closestPoint, minDist


