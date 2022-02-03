# Using PETSc to solve monolithic problem
import dolfin as d
import petsc4py.PETSc as PETSc
import ufl

class stubsSNESProblem():
    """To interface with PETSc SNES solver
        
    Notes on the high-level dolfin solver d.solve() when applied to Mixed Nonlinear problems:

    F is the sum of all forms, in stubs this is:
    Fsum = sum([f.lhs for f in model.forms]) # single form F0+F1+...+Fn
    d.solve(Fsum==0, u) roughly executes the following:


    * d.solve(Fsum==0, u)                                                       [fem/solving.py]
        * _solve_varproblem()                                                   [fem/solving.py]
            eq, ... = _extract_args()
            F = extract_blocks(eq.lhs) # tuple of forms (F0, F1, ..., Fn)       [fem/formmanipulations -> ufl/algorithms/formsplitter]
            for Fi in F:
                for uj in u._functions:
                    Js.append(expand_derivatives(formmanipulations.derivative(Fi, uj)))
                    # [J00, J01, J02, etc...]
            problem = MixedNonlinearVariationalProblem(F, u._functions, bcs, Js)
            solver  = MixedNonlinearVariationalSolver(problem)
            solver.solve()

    * MixedNonlinearVariationalProblem(F, u._functions, bcs, Js)     [fem/problem.py] 
        u_comps = [u[i]._cpp_object for i in range(len(u))] 

        # if len(F)!= len(u) -> Fill empty blocks of F with None

        # Check that len(J)==len(u)**2 and len(F)==len(u)

        # use F to create Flist. Separate forms by domain:
        # Flist[i] is a list of Forms separated by domain. E.g. if F1 consists of integrals on \Omega_1, \Omega_2, and \Omega_3
        # then Flist[i] is a list with 3 forms
        If Fi is None -> Flist[i] = cpp.fem.Form(1,0) 
        else -> Flist[i] = [Fi[domain=0], Fi[domain=1], ...]

        # Do the same for J -> Jlist

        cpp.fem.MixedNonlinearVariationalProblem.__init__(self, Flist, u_comps, bcs, Jlist)
        
    ========
    More notes:
    ========
    # on extract_blocks(F)
    F  = sum([f.lhs for f in model.forms]) # single form
    Fb = extract_blocks(F) # tuple of forms
    Fb0 = Fb[0]

    F0 = sum([f.lhs for f in model.forms if f.compartment.name=='cytosol'])
    F0.equals(Fb0) -> False

    I0 = F0.integrals()[0].integrand()
    Ib0 = Fb0.integrals()[0].integrand()
    I0.ufl_operands[0] == Ib0.ufl_operands[0] -> False (ufl.Indexed(Argument))) vs ufl.Indexed(ListTensor(ufl.Indexed(Argument)))
    I0.ufl_operands[1] == Ib0.ufl_operands[1] -> True
    I0.ufl_operands[0] == Ib0.ufl_operands[0](1) -> True


    # on d.functionspace
    V.__repr__() shows the UFL coordinate element (finite element over coordinate vector field) and finite element of the function space.
    We can access individually with:
    V.ufl_domain().ufl_coordinate_element()
    V.ufl_element()

    # on assembler
    d.fem.assembling.assemble_mixed(form, tensor)
    assembler = cpp.fem.MixedAssembler()


    fem.assemble.cpp/assemble_mixed(GenericTensor& A, const Form& a, bool add)
    MixedAssembler assembler;
    assembler.add_values = add;
    assembler.assemble(A, a);
    """


    def __init__(self, u, Fforms, Jforms):
        self.u = u
        self.Fforms = Fforms
        self.Jforms = Jforms

        # List of lists (lists partitioned by integration domains)
        assert isinstance(Fforms, list)
        assert isinstance(Fforms[0], list)
        assert isinstance(Jforms, list)
        assert isinstance(Jforms[0], list)
        assert isinstance(self.Jforms[0][0], (ufl.Form, d.Form))
        assert isinstance(self.Fforms[0][0], (ufl.Form,d.Form))

        self.dim = len(Fforms)
        assert len(Jforms) == self.dim**2

        # save sparsity patterns of block matrices
        self.tensors = [[None]*len(Jij_list) for Jij_list in self.Jforms]

    def initialize_petsc_matnest(self):
        dim = self.dim

        #Jdpetsc = [[None]*dim]*dim
        Jpetsc = []
        for i in range(dim):
            for j in range(dim):
                ij = i*dim + j
                #Jdpetsc[i][j] = d.PETScMatrix()
                Jsum = []
                Jsum = d.as_backend_type(d.assemble_mixed(self.Jforms[ij][0], tensor=d.PETScMatrix()))#, tensor=Jdpetsc[i][j])
                for k in range(1,len(self.Jforms[ij])):
                    Jsum += d.as_backend_type(d.assemble_mixed(self.Jforms[ij][k], tensor=d.PETScMatrix()))#, tensor=Jdpetsc[i][j])

                #Jsum.mat().assemble() 
                Jpetsc.append(Jsum)#Jdpetsc[i][j].mat()

        self.Jpetsc_nest = d.PETScNestMatrix(Jpetsc).mat()
        self.Jpetsc_nest.assemble() 
        #return Jpetsc_nest

 
    def initialize_petsc_vecnest(self):
        dim = self.dim

        Fpetsc = []
        for j in range(dim):
            Fsum = d.as_backend_type(d.assemble_mixed(self.Fforms[j][0]))#, tensor=Fdpetsc[j])
            for k in range(1,len(self.Fforms[j])):
                Fsum += d.as_backend_type(d.assemble_mixed(self.Fforms[j][k]))#, tensor=Fdpetsc[j])

            Fsum.vec().assemble()
            Fpetsc.append(Fsum.vec())
        
        self.Fpetsc_nest = PETSc.Vec().createNest(Fpetsc)
        self.Fpetsc_nest.assemble()
        #return Fpetsc_nest

    def assemble_Jnest(self, Jnest):
        """Assemble Jacobian nest matrix

        Parameters
        ----------
        Jnest : petsc4py.Mat
            PETSc nest matrix representing the Jacobian

        Jmats are created using assemble_mixed(Jform) and are dolfin.PETScMatrix types
        """
        dim = self.dim
        #Jmats = []
        # Get the petsc sub matrices, convert to dolfin wrapper, assemble forms using dolfin wrapper as tensor
        #for ij, Jij_forms in enumerate(self.Jforms):
        for i in range(dim):
            for j in range(dim):
                ij = i*dim+j
                Jij_petsc = Jnest.getNestSubMatrix(i,j)
                num_subforms = len(self.Jforms[ij])

                if num_subforms==1:
                    d.assemble_mixed(self.Jforms[ij][0], tensor=d.PETScMatrix(Jij_petsc))
                    #Jij_petsc.assemble()
                    continue
                else:
                    #Jmats.append([])
                    Jmats=[]
                    # Jijk == dFi/duj(Omega_k)
                    for k, Jijk_form in enumerate(self.Jforms[ij]):
                        # if we have the sparsity pattern re-use it, if not save it for next time
                        if self.tensors[ij][k] is None:
                            self.tensors[ij][k] = d.PETScMatrix()
                        Jmats.append(d.assemble_mixed(Jijk_form, tensor=self.tensors[ij][k]))
                        #Jmats[ij].append(d.assemble_mixed(Jijk_form, tensor=Jij_dolfin))
                        #Jmats[ij].append(d.assemble_mixed(Jijk_form, tensor=Jij_petsc)

                    # sum the matrices
                    Jij_petsc.zeroEntries() # this maintains sparse structure
                    for Jmat in Jmats:
                        # structure options: SAME_NONZERO_PATTERN, DIFFERENT_NONZERO_PATTERN, SUBSET_NONZERO_PATTERN, UNKNOWN_NONZERO_PATTERN 
                        Jij_petsc.axpy(1, d.as_backend_type(Jmat).mat(), structure=Jij_petsc.Structure.SUBSET_NONZERO_PATTERN) 
                    #     Jij_petsc.axpy(1, d.as_backend_type(Jmat).mat())
                    # assemble petsc
                    #Jij_petsc.assemble()    
        # assemble petsc
        Jnest.assemble()

    def assemble_Fnest(self, Fnest):
        dim = self.dim
        Fi_petsc = Fnest.getNestSubVecs()
        Fvecs = []
        for j in range(dim):
            Fvecs.append([])
            for k in range(len(self.Fforms[j])):
                Fvecs[j].append(d.as_backend_type(d.assemble_mixed(self.Fforms[j][k])))#, tensor=d.PETScVector(Fvecs[idx]))
            # sum the vectors
            Fi_petsc[j].zeroEntries()
            for k in range(len(self.Fforms[j])):
                Fi_petsc[j].axpy(1, Fvecs[j][k].vec())
        
        # assemble petsc
        for j in range(dim):
            Fi_petsc[j].assemble()
        Fnest.assemble()
            
    def copy_u(self, unest):
        uvecs = unest.getNestSubVecs()
        duvecs = [None]*self.dim
        for idx, uvec in enumerate(uvecs):
            uvec.copy(self.u.sub(idx).vector().vec())
            self.u.sub(idx).vector().apply("")
            # duvecs[idx] = d.PETScVector(uvec)   # convert petsc.Vec -> d.PETScVector
            # duvecs[idx].vec().copy(self.u.sub(idx).vector().vec())
            # self.u.sub(idx).vector().apply("")      

    def F(self, snes, u, Fnest):
        self.copy_u(u)
        self.assemble_Fnest(Fnest)

    def J(self, snes, u, Jnest, P):
        self.copy_u(u)
        self.assemble_Jnest(Jnest)





# """
# Solver classes. Linear/Nonlinear are wrappers around dolfin solvers.
# MultiphysicsSolver uses a picard iteration scheme to solve coupled problems.
# """
# import re
# import os
# from pandas import read_json
# import dolfin as d
# from stubs.common import nan_to_none
# from stubs.common import round_to_n
# from stubs import model_assembly
# import random
# import numpy as np
# from scipy.spatial.transform import Rotation as Rot

# import petsc4py.PETSc as PETSc
# Print = PETSc.Sys.Print

# import mpi4py.MPI as pyMPI
# comm = d.MPI.comm_world
# rank = comm.rank
# size = comm.size
# root = 0

# # Base Solver class
# class Solver:
#     def __init__(self, framework='dolfin'):
#         self.framework = framework

#     def check_validity(self):
#         if self.framework not in ['dolfin']:
#             raise ValueError(f"Framework {self.framework} is not supported.")

#         print('Solver settings are valid')

# # Base MultiphysicsSolver class
# class MultiphysicsSolver(Solver):
#     def __init__(self, method='iterative', eps_Fabs=1e-6, eps_udiff_abs=1e-8, eps_udiff_rel=1e-5,
#                  min_multiphysics=2, max_multiphysics=6, dt_increase_factor=1.0, dt_decrease_factor=0.8):
#         super().__init__()
#         self.method             = method
#         self.eps_Fabs           = eps_Fabs
#         self.eps_udiff_abs      = eps_udiff_abs
#         self.eps_udiff_rel      = eps_udiff_rel
#         self.min_multiphysics   = min_multiphysics
#         self.max_multiphysics   = max_multiphysics
#         self.dt_increase_factor = dt_increase_factor
#         self.dt_decrease_factor = dt_decrease_factor

#     def check_validity(self):
#         super().check_validity()
#         if self.method not in ['iterative', 'monolithic']:
#             raise ValueError(f"MultiphysicsSolver must be either 'iterative' or 'monolithic'.")

#         print('All MultiphysicsSolver settings are valid!')

# # Base NonlinearSolver class
# class NonlinearSolver(Solver):
#     def __init__(self, method='newton', min_nonlinear=2, max_nonlinear=10,
#                  dt_increase_factor=1.0, dt_decrease_factor=0.8,):
#         super().__init__()
#         self.method=method
#         self.min_nonlinear=min_nonlinear
#         self.max_nonlinear=max_nonlinear
#         self.dt_increase_factor=dt_increase_factor
#         self.dt_decrease_factor=dt_decrease_factor
#     def check_validity(self):
#         # check settings of parent class
#         super().check_validity()
#         """
#         Check that all current settings are valid
#         """
#         if type(self.min_nonlinear) != int or type(self.max_nonlinear) != int:
#             raise TypeError("min_nonlinear and max_nonlinear must be integers.")

#         if self.dt_increase_factor < 1.0: 
#             raise ValueError("dt_increase_factor must be >= 1.0")

#         print("All NonlinearSolver settings are valid!")

# class NonlinearNewtonSolver(NonlinearSolver):
#     """
#     Settings for dolfin nonlinear Newton solver
#     """
#     def __init__(self, maximum_iterations=50, error_on_nonconvergence=False, relative_tolerance=1e-6, 
#                  absolute_tolerance=1e-8, min_nonlinear=None, max_nonlinear=None, dt_increase_factor=None, dt_decrease_factor=None):
#         super().__init__(method = 'newton')

#         self.maximum_iterations         = maximum_iterations
#         self.error_on_nonconvergence    = error_on_nonconvergence
#         self.relative_tolerance         = relative_tolerance
#         self.absolute_tolerance         = absolute_tolerance

#         for prop_name, prop in {'min_nonlinear':min_nonlinear, 'max_nonlinear':max_nonlinear,
#                                 'dt_increase_factor':dt_increase_factor, 'dt_decrease_factor': dt_decrease_factor}.items():
#             if prop is not None:
#                 setattr(self, prop_name, prop)

#     def check_validity(self):
#         super().check_validity()
#         assert hasattr(self, 'framework')
#         assert self.framework in ['dolfin']
#         if type(self.maximum_iterations) != int:
#             raise TypeError("maximum_iterations must be an int")
#         if self.relative_tolerance <= 0: 
#             raise ValueError("relative_tolerance must be greater than 0")
#         if self.absolute_tolerance <= 0: 
#             raise ValueError("absolute_tolerance must be greater than 0")

#         print("All NonlinearNewtonSolver settings are valid!")

# class NonlinearPicardSolver(NonlinearSolver):
#     def __init__(self, picard_norm = 'Linf', maximum_iterations=50):
#         super().__init__(method='picard')

#         self.maximum_iterations         = maximum_iterations
#         self.picard_norm = picard_norm

#     def check_validity(self):
#         super().check_validity()
#         valid_norms = ['Linf', 'L2']
#         if picard_norm not in valid_norms:
#             raise ValueError(f"Invalid Picard norm ({picard_norm}) reasonable values are: {valid_norms}")

# # Base LinearSolver class
# class LinearSolver(Solver):
#     def __init__(self, method='gmres', preconditioner='hypre_amg'):
#         super().__init__()
#         self.method         = method
#         self.preconditioner = preconditioner
#     def check_validity(self):
#         super().check_validity()

# class DolfinKrylovSolver(LinearSolver):
#     def __init__(self, method = 'gmres', preconditioner = 'hypre_amg', 
#                  maximum_iterations = 100000, error_on_nonconvergence = False,
#                  nonzero_initial_guess = True, relative_tolerance = 1e-6,
#                  absolute_tolerance = 1e-8):
#         super().__init__(method = method, preconditioner = preconditioner)
#         self.maximum_iterations         = maximum_iterations
#         self.error_on_nonconvergence    = error_on_nonconvergence
#         self.nonzero_initial_guess      = nonzero_initial_guess
#         self.relative_tolerance         = relative_tolerance
#         self.absolute_tolerance         = absolute_tolerance

#     def check_validity(self):
#         super().check_validity()
#         if self.method not in d.krylov_solver_methods().keys():
#             raise ValueError(f"Method {self.method} is not a supported dolfin krylov solver.")
#         if self.preconditioner not in d.krylov_solver_preconditioners().keys():
#             raise ValueError(f"Preconditioner {self.preconditioner} is not a supported dolfin krylov preconditioner.")


# class SolverSystem:
#     def __init__(self, final_t, initial_dt, multiphysics_solver=MultiphysicsSolver(), 
#                  nonlinear_solver=NonlinearNewtonSolver(), linear_solver=DolfinKrylovSolver(), 
#                  ignore_surface_diffusion=False, auto_preintegrate=True,
#                  adjust_dt=None):

#         self.final_t                    = final_t
#         self.initial_dt                 = initial_dt
#         self.multiphysics_solver        = multiphysics_solver
#         self.nonlinear_solver           = nonlinear_solver
#         self.linear_solver              = linear_solver

#         # Sort adjust_dt if it was provided
#         if adjust_dt is not None:
#             self.check_adjust_dt_validity(adjust_dt)
#             adjust_dt.sort(key = lambda tuple_: tuple_[0]) # sort by the first index (time to reset)
#         self.adjust_dt                  = adjust_dt

#         # General settings 
#         self.ignore_surface_diffusion   = ignore_surface_diffusion # setting to True will treat surface variables as ODEs 
#         self.auto_preintegrate          = auto_preintegrate

#     def check_solver_system_validity(self):
#         if self.final_t <= 0:
#             raise ValueError("final_t must be > 0")
#         if self.initial_dt <= 0:
#             raise ValueError("initial_dt must be > 0")
#         if self.initial_dt > self.final_t:
#             raise ValueError("initial_dt must be < final_t")
            

#         self.check_adjust_dt_validity(self.adjust_dt)
#         if self.adjust_dt is not None and self.adjust_dt != sorted(self.adjust_dt, key = lambda tuple_: tuple_[0]):
#             raise ValueError("adjust_dt not sorted...")

#         for solver in [self.multiphysics_solver, self.nonlinear_solver, self.linear_solver]:
#             if solver is not None:
#                 solver.check_validity()
#             else:
#                 Print(f"Warning: SolverSystem does not include a {type(solver).__name__}.")

#     def check_adjust_dt_validity(self, adjust_dt):
#         # make sure adjust_dt is a list of 2-tuples with proper values
#         # (time to reset, new dt)
#         if adjust_dt is not None:
#             if type(adjust_dt) is not list:
#                 raise ValueError("adjust_dt must be a list of 2-tuples")
#             if len(adjust_dt) >= 1:
#                 for tuple_ in adjust_dt:
#                     if type(tuple_) != tuple or len(tuple_) != 2:
#                         raise ValueError("adjust_dt must be a list of 2-tuples")
#                     if tuple_[0] < 0:
#                         raise ValueError("First index of 2-tuples in adjust_dt (time to reset) must be >= 0")

#     def make_dolfin_parameter_dict(self):
#         """
#         Helper function to package all necessary parameters into dictionaries which can be input as dolfin solver parameters
#         """
#         nonlinear_solver_keys = ['maximum_iterations', 'error_on_nonconvergence', 'absolute_tolerance', 'relative_tolerance']
#         self.nonlinear_dolfin_solver_settings = {**{k:v for k,v in self.nonlinear_solver.__dict__.items() if k in nonlinear_solver_keys}, 
#                                   **{'linear_solver': self.linear_solver.method,
#                                      'preconditioner': self.linear_solver.preconditioner}}
#         linear_solver_keys = ['maximum_iterations', 'error_on_nonconvergence', 'nonzero_initial_guess', 'absolute_tolerance', 'relative_tolerance']
#         self.linear_dolfin_solver_settings = {**{k:v for k,v in self.linear_solver.__dict__.items() if k in linear_solver_keys}}
