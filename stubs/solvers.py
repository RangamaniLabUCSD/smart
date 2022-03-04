# # Using PETSc to solve monolithic problem
import dolfin as d
import petsc4py.PETSc as PETSc
import ufl
from stubs.common import _fancy_print as fancy_print
import time


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


    def __init__(self, u, Fforms, Jforms, active_compartments, all_compartments, stopwatches, mpi_comm_world):
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

        self.mpi_comm_world = mpi_comm_world

        # save sparsity patterns of block matrices
        self.tensors = [[None]*len(Jij_list) for Jij_list in self.Jforms]

        # Need block sizes because some forms may be empty
        self.block_sizes = [c._num_dofs for c in active_compartments]
        self.is_single_domain = len(self.block_sizes) == 1

        self.active_compartment_names = [c.name for c in active_compartments]
        self.mesh_id_to_name = {c.mesh_id:c.name for c in all_compartments}

        # Timings
        self.stopwatches = stopwatches

    def initialize_petsc_matnest(self):
        dim = self.dim
        fancy_print(f"Initializing block Jacobian", format_type='assembly')

        #Jdpetsc = [[None]*dim]*dim
        Jpetsc = []
        for i in range(dim):
            for j in range(dim):
                ij = i*dim + j

                Jsum = None
                for k in range(len(self.Jforms[ij])):
                    # compartment names for indices
                    if self.Jforms[ij][k].function_space(0) is None:
                        fancy_print(f"{self.Jijk_name(i,j,k=None)} has no function space", format_type='log')
                        continue

                    # initialize the tensor
                    if self.tensors[ij][k] is None:
                        self.tensors[ij][k] = d.PETScMatrix()
                    if Jsum is None:
                        Jsum = d.as_backend_type(d.assemble_mixed(self.Jforms[ij][k], tensor=self.tensors[ij][k]))
                    else:
                        Jsum += d.as_backend_type(d.assemble_mixed(self.Jforms[ij][k], tensor=self.tensors[ij][k]))

                    # FIXME - not printing the correct domain 
                    # (need to extract from form.function_space.mesh().id())
                    fancy_print(f"Initialized {self.Jijk_name(i,j,k)}, tensor size = {Jsum.size(0), Jsum.size(1)}", format_type='log')
                if Jsum is None:
                    fancy_print(f"{self.Jijk_name(i,j)} is empty - initializing as empty PETSc Matrix with size {self.block_sizes[i]}, {self.block_sizes[j]}", format_type='log')
                    Jsum = self.init_zero_petsc_matrix(self.block_sizes[i], self.block_sizes[j])
                    #raise AssertionError()
                    # tensor = d.PETScMatrix(Jsum.size(0))

                #Jsum.mat().assemble() 
                Jpetsc.append(Jsum)#Jdpetsc[i][j].mat()

        if self.is_single_domain == 1:
            # We can't use a nest matrix
            self.Jpetsc_nest = Jpetsc[0].mat() #d.PETScMatrix(Jpetsc[0]).mat()
        else:
            self.Jpetsc_nest = d.PETScNestMatrix(Jpetsc).mat()
        self.Jpetsc_nest.assemble()
        # debugging
        # self.Jpetsc_init = Jpetsc
        # return Jpetsc

    def initialize_petsc_vecnest(self):
        dim = self.dim
        fancy_print(f"Initializing block residual vector", format_type='assembly')

        Fpetsc = []
        for j in range(dim):
            # Fsum = d.as_backend_type(d.assemble_mixed(self.Fforms[j][0]))#, tensor=Fdpetsc[j])
            # for k in range(1,len(self.Fforms[j])):
            #     Fsum += d.as_backend_type(d.assemble_mixed(self.Fforms[j][k]))#, tensor=Fdpetsc[j])

            Fsum = None
            for k in range(len(self.Fforms[j])):
                if self.Fforms[j][k].function_space(0) is None:
                    fancy_print(f"{self.Fjk_name(j,k)}] has no function space", format_type='log')
                    continue
                if Fsum is None:
                    Fsum = d.as_backend_type(d.assemble_mixed(self.Fforms[j][k], tensor=d.PETScVector()))
                else:
                    Fsum += d.as_backend_type(d.assemble_mixed(self.Fforms[j][k], tensor=d.PETScVector()))
            if Fsum is None:
                fancy_print(f"{self.Fjk_name(j)} is empty - initializing as empty PETSc Vector with size {self.block_sizes[j]}")
                Fsum = self.init_zero_petsc_vector(self.block_sizes[j])
                #raise AssertionError()

            # Fsum.vec().assemble()
            Fpetsc.append(Fsum.vec())
        
        if self.is_single_domain == 1:
            # We can't use a nest vector
            self.Fpetsc_nest = d.PETScVector(Fpetsc[0]).vec()
        else:
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
        fancy_print(f"Assembling block Jacobian", format_type='assembly')
        self.stopwatches["snes jacobian assemble"].start()
        dim = self.dim

        # Check for empty forms
        empty_forms = []
        for i in range(dim):
            for j in range(dim):
                ij = i*dim+j
                if all(self.Jforms[ij][k].function_space(0) is None for k in range(len(self.Jforms[ij]))):
                    empty_forms.append((i,j))
        if len(empty_forms) > 0:
            fancy_print(f"Forms {empty_forms} are empty. Skipping assembly.", format_type='data')

        # Get the petsc sub matrices, convert to dolfin wrapper, assemble forms using dolfin wrapper as tensor
        #for ij, Jij_forms in enumerate(self.Jforms):
        for i in range(dim):
            for j in range(dim):
                if (i,j) in empty_forms:
                    continue
                ij = i*dim+j
                num_subforms = len(self.Jforms[ij])
                
                # Extract petsc submatrix
                if self.is_single_domain == 1:
                    Jij_petsc = Jnest
                else:
                    Jij_petsc = Jnest.getNestSubMatrix(i,j)

                if num_subforms==1 and self.Jforms[ij][0].function_space(0) is None:
                    continue
                fancy_print(f"Assembling {self.Jijk_name(i,j)}:", format_type='assembly_sub')

                # Assemble the form
                if num_subforms==1:
                    # Check for empty form
                    if self.Jforms[ij][0].function_space(0) is not None:
                        d.assemble_mixed(self.Jforms[ij][0], tensor=d.PETScMatrix(Jij_petsc))
                        self.print_Jijk_info(i,j,k=None,tensor=Jij_petsc)
                        continue
                    else:
                        raise AssertionError()
                else:
                    Jmats=[]
                    # Jijk == dFi/duj(Omega_k)
                    for k in range(num_subforms):
                        # Check for empty form
                        if self.Jforms[ij][k].function_space(0) is None:
                            fancy_print(f"{self.Jijk_name(i,j,k)} is empty. Skipping assembly.", format_type='data')
                            continue
                        # if we have the sparsity pattern re-use it, if not save it for next time
                        # single domain can't re-use the tensor for some reason
                        if self.tensors[ij][k] is None or self.is_single_domain: 
                            self.tensors[ij][k] = d.PETScMatrix()
                        else:
                            fancy_print(f"Reusing tensor for {self.Jijk_name(i,j,k)}", format_type='data')
                        # Assemble and append to the list of subforms
                        Jmats.append(d.assemble_mixed(self.Jforms[ij][k], tensor=self.tensors[ij][k]))
                        # Print some useful info on assembled Jijk
                        self.print_Jijk_info(i,j,k,tensor=self.tensors[ij][k].mat())

                # Sum the assembled forms
                Jij_petsc.zeroEntries() # this maintains sparse (non-zeros) structure
                for Jmat in Jmats:
                    # structure options: SAME_NONZERO_PATTERN, DIFFERENT_NONZERO_PATTERN, SUBSET_NONZERO_PATTERN, UNKNOWN_NONZERO_PATTERN 
                    Jij_petsc.axpy(1, d.as_backend_type(Jmat).mat(), structure=Jij_petsc.Structure.SUBSET_NONZERO_PATTERN) 
                #Jij_petsc.assemble()    

                self.print_Jijk_info(i,j,k=None,tensor=Jij_petsc)

        # assemble petsc
        Jnest.assemble()

        self.stopwatches["snes jacobian assemble"].pause()

    def assemble_Fnest(self, Fnest):
        dim = self.dim
        fancy_print(f"Assembling block residual vector", format_type='assembly')
        self.stopwatches["snes residual assemble"].start()

        if self.is_single_domain:
            Fj_petsc = [Fnest]
        else:
            Fj_petsc = Fnest.getNestSubVecs()
        Fvecs = []

        for j in range(dim):
            Fvecs.append([])
            for k in range(len(self.Fforms[j])):
                Fvecs[j].append(d.as_backend_type(d.assemble_mixed(self.Fforms[j][k])))#, tensor=d.PETScVector(Fvecs[idx]))
            # TODO: could probably speed this up by not using axpy if there is only one subform
            # sum the vectors
            Fj_petsc[j].zeroEntries()
            for k in range(len(self.Fforms[j])):
                Fj_petsc[j].axpy(1, Fvecs[j][k].vec())
        
        # assemble petsc
        # for j in range(dim):
        #     Fi_petsc[j].assemble()
        Fnest.assemble()
        self.stopwatches["snes residual assemble"].pause()
            
    def copy_u(self, unest):
        if self.is_single_domain:
            uvecs = [unest]
        else:
            uvecs = unest.getNestSubVecs()

        for idx, uvec in enumerate(uvecs):
            uvec.copy(self.u.sub(idx).vector().vec())
            self.u.sub(idx).vector().apply("")

    def F(self, snes, u, Fnest):
        self.copy_u(u)
        self.assemble_Fnest(Fnest)

    def J(self, snes, u, Jnest, P):
        self.copy_u(u)
        self.assemble_Jnest(Jnest)

    def init_zero_petsc_matrix(self, dim0, dim1):
        """Initialize a dolfin wrapped PETSc matrix with all zeros

        Parameters
        ----------
        dim : int
            Size of matrix
        """
        # M = PETSc.Mat().create()
        # M.setSizes([dim, dim])
        # M.setType("aij")
        # M.setUp()
        self.stopwatches['snes initialize zero matrices'].start()
        M = PETSc.Mat().createAIJ(size=(dim0,dim1), nnz=0, comm=self.mpi_comm_world)
        M.assemble()
        self.stopwatches['snes initialize zero matrices'].pause()
        return d.PETScMatrix(M)

    def init_zero_petsc_vector(self, dim0):
        """Initialize a dolfin wrapped PETSc vector with all zeros

        Parameters
        ----------
        dim0 : int
            Size of vector
        """

        V = PETSc.Vec().createSeq(dim0, comm=self.mpi_comm_world)
        V.assemble()
        return d.PETScVector(V)

    def Jijk_name(self, i, j, k=None):
        ij = i*self.dim + j
        if k is None:
            return f"J{i}{j} = dF[{self.active_compartment_names[i]}]/du[{self.active_compartment_names[j]}]"
        else:
            domain_name = self.mesh_id_to_name[self.Jforms[ij][k].function_space(0).mesh().id()]
            return f"J{i}{j}{k} = dF[{self.active_compartment_names[i]}]/du[{self.active_compartment_names[j]}] (domain={domain_name})"
    
    def Fjk_name(self, j, k=None):
        if k is None:
            return f"F{j} = F[{self.active_compartment_names[j]}]"
        else:
            domain_name = self.mesh_id_to_name[self.Fforms[j][k].function_space(0).mesh().id()]
            return f"F{j} = F[{self.active_compartment_names[j]}] (domain={domain_name})"
            
    def print_Jijk_info(self, i, j, k=None, tensor=None):
        if tensor is None:
            return
        # Print some useful info on Jijk
        info = tensor.getInfo()
        # , block_size={int(info['block_size'])}
        info_str = f"size={str(tensor.size)[1:-1]: <18}, nnz={int(info['nz_allocated']): <8}, memory[MB]={int(1e-6*info['memory']): <6}, "\
                    f"assemblies={int(info['assemblies']): <4}, mallocs={int(info['mallocs']): <4}\n"
        if k is None:
            fancy_print(f"Assembled form {self.Jijk_name(i,j,k)}:\n{info_str}", format_type='data')
        else:
            fancy_print(f"Assembled subform {self.Jijk_name(i,j,k)}:\n{info_str}", format_type='data')
        if info['nz_unneeded'] > 0:
            fancy_print(f"WARNING: {info['nz_unneeded']} nonzero entries are unneeded", format_type='warning')

    def get_csr_matrix(self,i,j):
        "This is a matrix that can be used to visualize the sparsity pattern using plt.spy()"
        if self.is_single_domain:
            M = self.Jpetsc_nest
        else:
            M = self.Jpetsc_nest.getNestSubMatrix(i,j)
        from scipy.sparse import csr_matrix
        return csr_matrix(M.getValuesCSR()[::-1], shape=M.size) 

