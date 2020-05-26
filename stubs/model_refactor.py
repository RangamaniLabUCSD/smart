import dolfin as d
from collections import defaultdict as ddict
import petsc4py.PETSc as PETSc
Print = PETSc.Sys.Print
from termcolor import colored

import stubs
import stubs.model_assembly
import stubs.common as common
from stubs import unit as ureg
color_print = common.color_print

comm = d.MPI.comm_world
rank = comm.rank
size = comm.size
root = 0

class ModelRefactor(object):
    def __init__(self, PD, SD, CD, RD, config, solver_system, mesh=None):
        self.PD = PD
        self.SD = SD
        self.CD = CD
        self.RD = RD
        self.config = config
        # Check that solver_system is valid
        solver_system.check_solver_system_validity()
        self.solver_system = solver_system
        self.solver_system.make_dolfin_parameter_dict()

        self.mesh = mesh
        self.params = ddict(list)

        self.idx = 0
        self.NLidx = 0 # nonlinear iterations
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

        self.data = stubs.data_manipulation.Data(self)


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
        setattr(self.CD, 'meshes', {self.mesh.name: self.mesh.mesh})
        self.CD.extract_submeshes('cyto', save_to_file=False)
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
        self.SD.assign_initial_conditions()
        self.u = self.SD.u
        self.v = self.SD.v
        self.V = self.SD.V

        print("\n\n********** Model initialization (Part 6/6) **********")
        print("Assembling reactive and diffusive fluxes...\n")
        self.RD.reaction_to_fluxes()
        self.RD.do_to_all('reaction_to_fluxes')
        self.FD = self.RD.get_flux_container()
        self.FD.do_to_all('get_additional_flux_properties', {"CD": self.CD, "solver_system": self.solver_system})
        self.FD.do_to_all('flux_to_dolfin')
 
        self.set_allow_extrapolation()
        # Turn fluxes into fenics/dolfin expressions
        self.assemble_reactive_fluxes()
        self.assemble_diffusive_fluxes()
        self.sort_forms()

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

                if sp.dimensionality == max_dim and not sp.parent_species:
                    dx = sp.compartment.dx
                    Dform = D*d.inner(d.grad(u), d.grad(v)) * dx
                    self.Forms.add(stubs.model_assembly.Form(Dform, sp, 'D'))
                elif sp.dimensionality < max_dim or sp.parent_species:
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
            Print("Splitting problem into bilinear and linear forms for picard iterations: a(u,v) == L(v)")
            for comp in comp_list:
                comp_forms = [f.dolfin_form for f in self.Forms.select_by('compartment_name', comp.name)]
                self.a[comp.name] = d.lhs(sum(comp_forms))
                self.L[comp.name] = d.rhs(sum(comp_forms))
                problem = d.LinearVariationalProblem(self.a[comp.name],
                                                     self.L[comp.name], self.u[comp.name]['u'], [])
                self.linear_solver[comp.name] = d.LinearVariationalSolver(problem)
                p = self.linear_solver[comp.name].parameters
                p['linear_solver'] = self.solver_system.linear_solver.method
                if type(self.solver_system.linear_solver) == stubs.solvers.DolfinKrylovSolver:
                    p['krylov_solver'].update(self.solver_system.linear_solver.__dict__)
                    p['krylov_solver'].update({'nonzero_initial_guess': True}) # important for time dependent problems

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
    def solve(self, op_split_scheme = "DRD", plot_period=1):
        ## solve
        self.init_solver_and_plots()

        self.stopwatch("Total simulation")
        while True:
            # Solve using specified operator-splitting scheme (just DRD for now)
            if op_split_scheme == "DRD":
                self.DRD_solve(boundary_method='RK45')
            elif op_split_scheme == "DR":
                self.DR_solve(boundary_method='RK45')
            else:
                raise Exception("I don't know what operator splitting scheme to use")

            self.compute_statistics()
            if self.idx % plot_period == 0 or self.t >= self.final_t:
                self.plot_solution()
                self.plot_solver_status()
            if self.t >= self.final_t:
                break

        self.stopwatch("Total simulation", stop=True)
        Print("Solver finished with %d total time steps." % self.idx)


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


    def forward_time_step(self, factor=1):

        self.dT.assign(float(self.dt*factor))
        self.t = float(self.t+self.dt*factor)
        self.T.assign(self.t)
        #print("t: %f , dt: %f" % (self.t, self.dt*factor))

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
        if not t:
            t = self.t
        if t0 and dt:
            raise Exception("Specify either t0 or dt, not both.")
        elif t0 != None:
            dt = t-t0
        elif dt != None:
            t0 = t-dt
        if t0 != None:
            if t0<0 or dt<0:
                raise Exception("Either t0 or dt is less than 0, is this the desired behavior?")

        # Update time dependent parameters
        for param_name, param in self.PD.Dict.items():
            # check to make sure a parameter wasn't assigned a new value more than once
            value_assigned = 0
            if not param.is_time_dependent:
                continue
            # use value by evaluating symbolic expression
            if param.symExpr and not param.preintegrated_symExpr:
                newValue = param.symExpr.subs({'t': t}).evalf()
                value_assigned += 1

            # calculate a preintegrated expression by subtracting previous value
            # and dividing by time-step
            if param.symExpr and param.preintegrated_symExpr:
                if t0 == None:
                    raise Exception("Must provide a time interval for"\
                                    "pre-integrated variables.")
                newValue = (param.preintegrated_symExpr.subs({'t': t}).evalf()
                            - param.preintegrated_symExpr.subs({'t': t0}).evalf())/dt
                value_assigned += 1

            # if parameter is given by data
            if param.sampling_data is not None and param.preint_sampling_data is None:
                data = param.sampling_data
                # We never want time to extrapolate beyond the provided data.
                if t<data[0,0] or t>data[-1,0]:
                    raise Exception("Parameter cannot be extrapolated beyond"\
                                    "provided data.")
                # Just in case... return a nan if value is outside of bounds
                newValue = np.interp(t, data[:,0], data[:,1],
                                     left=np.nan, right=np.nan)
                value_assigned += 1

            # if parameter is given by data and it has been pre-integrated
            if param.sampling_data is not None and param.preint_sampling_data is not None:
                int_data = param.preint_sampling_data
                oldValue = np.interp(t0, int_data[:,0], int_data[:,1],
                                     left=np.nan, right=np.nan)
                newValue = (np.interp(t, int_data[:,0], int_data[:,1],
                                     left=np.nan, right=np.nan) - oldValue)/dt
                value_assigned += 1

            if value_assigned != 1:
                raise Exception("Either a value was not assigned or more than"\
                                "one value was assigned to parameter %s" % param.name)

            param.value = newValue
            param.dolfinConstant.assign(newValue)
            Print('%f assigned to time-dependent parameter %s'
                  % (newValue, param.name))
            self.params[param_name].append((t,newValue))


    def reset_timestep(self, comp_list=[]):
        """
        Resets the time back to what it was before the time-step. Optionally, input a list of compartments
        to have their function values reset (['n'] value will be assigned to ['u'] function).
        """
        self.set_time(self.t - self.dt, self.dt*self.solver_system.nonlinear_solver.dt_decrease_factor)
        Print("Resetting time-step and decreasing step size")
        for comp_name in comp_list:
            self.u[comp_name]['u'].assign(self.u[comp_name]['n'])
            Print("Assigning old value of u to species in compartment %s" % comp_name)

    def update_solution_boundary_to_volume(self):
        for comp_name in self.CD.Dict.keys():
            for key in self.u[comp_name].keys():
                if key[0] == 'v': # fixme
                    self.u[comp_name][key].interpolate(self.u[comp_name]['u'])
                    parent_comp_name = key[1:]
                    Print("Projected values from surface %s to volume %s" % (comp_name, parent_comp_name))

    def update_solution_volume_to_boundary(self):
        for comp_name in self.CD.Dict.keys():
            for key in self.u[comp_name].keys():
                if key[0] == 'b': # fixme
                    self.u[comp_name][key].interpolate(self.u[comp_name]['u'])
                    sub_comp_name = key[1:]
                    Print("Projected values from volume %s to surface %s" % (comp_name, sub_comp_name))


    def boundary_reactions_forward(self, factor=1, bcs=[], key='n'):
        self.stopwatch("Boundary reactions forward")
        nsubsteps = 1#int(self.config.solver['reaction_substeps'])

        for n in range(nsubsteps):
            self.forward_time_step(factor=factor/nsubsteps)
            self.updateTimeDependentParameters(t0=self.t)
            for comp_name, comp in self.CD.Dict.items():
                if comp.dimensionality < self.CD.max_dim:
                    self.nonlinear_solve(comp_name, factor=factor)
                    self.set_time(self.t-self.dt)
                    self.u[comp_name][key].assign(self.u[comp_name]['u'])
        self.stopwatch("Boundary reactions forward", stop=True)

    # def diffusion_forward(self, comp_name, factor=1, bcs=[]):
    #     self.stopwatch("Diffusion step ["+comp_name+"]")
    #     t0 = self.t
    #     self.forward_time_step(factor=factor)
    #     self.updateTimeDependentParameters(t0=t0)
    #     if self.config.solver['nonlinear'] == 'picard':
    #         self.picard_loop(comp_name, bcs)
    #     elif self.config.solver['nonlinear'] == 'newton':
    #         self.newton_iter(comp_name)
    #     self.u[comp_name]['n'].assign(self.u[comp_name]['u'])
    #     self.stopwatch("Diffusion step ["+comp_name+"]", stop=True)


    def nonlinear_solve(self, comp_name, factor=1.0):
        """
        A switch for choosing a nonlinear solver
        """
        if self.solver_system.nonlinear_solver.method == 'newton':
            self.NLidx, success = self.newton_iter(comp_name, factor=factor)
        elif self.solver_system.nonlinear_solver.method == 'picard':
            self.NLidx, success = self.picard_loop(comp_name, factor=factor)

        return self.NLidx, success


    def DRD_solve(self, bcs=[], boundary_method='RK45'):
        """
        General DRD operator splitting. Can be used with different non-linear
        solvers
        """
        Print('\n\n\n')
        self.idx += 1
        self.check_dt_resets()
        color_print('\n *** Beginning time-step %d [time=%f, dt=%f] ***\n' % (self.idx, self.t, self.dt), color='red')

        self.stopwatch("Total time step")

        # solve volume problem(s)
        for comp_name, comp in self.CD.Dict.items():
            if comp.dimensionality == self.CD.max_dim:
                #self.NLidx, success = self.newton_iter(comp_name, factor=0.5)
                self.NLidx, success = self.nonlinear_solve(comp_name, factor=0.5)
                self.set_time(self.t-self.dt/2) # reset time back to t=t0
        self.update_solution_volume_to_boundary()

        # single iteration
        # solve boundary problem(s)
        self.boundary_reactions_forward(factor=1) # t from [t0, t+dt]. automatically resets time back to t0
        self.update_solution_boundary_to_volume()
        self.set_time(self.t-self.dt/2) # perform the second half-step

        # solve volume problem(s)
        for comp_name, comp in self.CD.Dict.items():
            if comp.dimensionality == self.CD.max_dim:
                self.NLidx, success = self.nonlinear_solve(comp_name, factor=0.5)
                self.set_time(self.t-self.dt/2)
        self.set_time(self.t+self.dt/2)
        self.update_solution_volume_to_boundary()

        # check if time step should be changed
        if self.NLidx <= self.solver_system.nonlinear_solver.min_nonlinear:
            self.set_time(self.t, dt=self.dt*self.solver_system.nonlinear_solver.dt_increase_factor)
            Print("Increasing step size")
        if self.NLidx > self.solver_system.nonlinear_solver.max_nonlinear:
            self.set_time(self.t, dt=self.dt*self.solver_system.nonlinear_solver.dt_decrease_factor)
            Print("Decreasing step size")

        self.stopwatch("Total time step", stop=True, color='cyan')

    def DR_solve(self, bcs=[], boundary_method='RK45'):
        """
        General DR operator splitting. Can be used with different non-linear
        solvers
        """
        Print('\n\n\n')
        self.idx += 1
        self.check_dt_resets()
        color_print('\n *** Beginning time-step %d [time=%f, dt=%f] ***\n' % (self.idx, self.t, self.dt), color='red')

        self.stopwatch("Total time step")

        # solve volume problem(s)
        for comp_name, comp in self.CD.Dict.items():
            if comp.dimensionality == self.CD.max_dim:
                self.NLidx, success = self.nonlinear_solve(comp_name, factor=1.0)
                self.set_time(self.t-self.dt) # reset time back to t=t0
        self.update_solution_volume_to_boundary()

        # single iteration
        # solve boundary problem(s)
        self.boundary_reactions_forward(factor=1) # t from [t0, t+dt]. automatically resets time back to t0
        self.update_solution_boundary_to_volume()

        # check if time step should be changed
        if self.NLidx <= self.solver_system.nonlinear_solver.min_nonlinear:
            self.set_time(self.t, dt=self.dt*self.solver_system.nonlinear_solver.dt_increase_factor)
            Print("Increasing step size")
        if self.NLidx > self.solver_system.nonlinear_solver.max_nonlinear:
            self.set_time(self.t, dt=self.dt*self.solver_system.nonlinear_solver.dt_decrease_factor)
            Print("Decreasing step size")

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

    def newton_iter(self, comp_name, factor=1, bcs=[], assign_to_n=True):
        """
        A single iteration of Newton's method for a single component.
        """
        self.stopwatch("Newton's method [%s]" % comp_name)
        t0 = self.t
        self.forward_time_step(factor=factor) # increment time afterwards
        self.updateTimeDependentParameters(t0=t0)

        idx, success = self.nonlinear_solver[comp_name].solve()

        if assign_to_n:
            self.u[comp_name]['n'].assign(self.u[comp_name]['u'])

        self.stopwatch("Newton's method [%s]" % comp_name, stop=True)
        Print("%d Newton iterations required for convergence." % idx)
        return idx, success

    def picard_loop(self, comp_name, factor=1, bcs=[], assign_to_n=True):
        """
        Continue picard iterations until a specified tolerance or count is
        reached.
        """
        self.stopwatch("Picard loop [%s]" % comp_name)
        t0 = self.t
        self.forward_time_step(factor=factor) # increment time afterwards
        self.updateTimeDependentParameters(t0=t0)
        self.pidx = 0 # count the number of picard iterations
        success = True

        # main loop
        while True:
            self.pidx += 1
            #linear_solver_settings = self.config.dolfin_krylov_solver

            # solve
            self.linear_solver[comp_name].solve()
            #d.solve(self.a[comp_name]==self.L[comp_name], self.u[comp_name]['u'], bcs, solver_parameters=linear_solver_settings)

            # update temporary value of u
            self.data.computeError(self.u, comp_name, self.solver_system.nonlinear_solver.picard_norm)
            self.u[comp_name]['k'].assign(self.u[comp_name]['u'])

            # Exit if error tolerance or max iterations is reached
            Print('Linf norm (%s) : %f ' % (comp_name, self.data.errors[comp_name]['Linf']['abs'][-1]))
            if self.data.errors[comp_name]['Linf']['abs'][-1] < self.solver_system.nonlinear_solver.absolute_tolerance:
                #print("Norm (%f) is less than linear_abstol (%f), exiting picard loop." %
                 #(self.data.errors[comp_name]['Linf'][-1], self.config.solver['linear_abstol']))
                break
#            if self.data.errors[comp_name]['Linf']['rel'][-1] < self.config.solver['linear_reltol']:
#                print("Norm (%f) is less than linear_reltol (%f), exiting picard loop." %
#                (self.data.errors[comp_name]['Linf']['rel'][-1], self.config.solver['linear_reltol']))
#                break

            if self.pidx >= self.solver_system.nonlinear_solver.maximum_iterations:
                Print("Max number of picard iterations reached (%s), exiting picard loop with abs error %f." %
                (comp_name, self.data.errors[comp_name]['Linf']['abs'][-1]))
                success = False
                break

        self.stopwatch("Picard loop [%s]" % comp_name, stop=True)

        if assign_to_n:
            self.u[comp_name]['n'].assign(self.u[comp_name]['u'])

        Print("%d Picard iterations required for convergence." % self.pidx)
        return self.pidx, success


#===============================================================================
#===============================================================================
# POST-PROCESSING
#===============================================================================
#===============================================================================
    def init_solver_and_plots(self):
        self.data.initSolutionFiles(self.SD, write_type='xdmf')
        self.data.storeSolutionFiles(self.u, self.t, write_type='xdmf')
        self.data.computeStatistics(self.u, self.t, self.dt, self.SD, self.PD, self.CD, self.FD, self.NLidx)
        self.data.initPlot(self.config, self.SD, self.FD)

    def update_solution(self):
        for key in self.u.keys():
            self.u[key]['n'].assign(self.u[key]['u'])
            self.u[key]['k'].assign(self.u[key]['u'])

    def compute_statistics(self):
        self.data.computeStatistics(self.u, self.t, self.dt, self.SD, self.PD, self.CD, self.FD, self.NLidx)
        # debug
        #self.data.computeProbeValues(self.u, self.t, self.dt, self.SD, self.PD, self.CD, self.FD, self.NLidx)
        self.data.outputPickle(self.config)
        self.data.outputCSV(self.config)

    def plot_solution(self):
        self.data.storeSolutionFiles(self.u, self.t, write_type='xdmf')
        self.data.plotParameters(self.config)
        self.data.plotSolutions(self.config, self.SD)
        self.data.plotFluxes(self.config)

    def plot_solver_status(self):
        self.data.plotSolverStatus(self.config)
