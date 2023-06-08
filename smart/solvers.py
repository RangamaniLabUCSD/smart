"""SMART module interfacing with PETSc SNES solver"""
import logging
from typing import Dict, List, Optional

import dolfin as d
import petsc4py.PETSc as p

from .common import Stopwatch
from .model_assembly import Compartment

logger = logging.getLogger(__name__)

__all__ = ["smartSNESProblem"]

_stopwatch_keys = [
    "snes jacobian assemble",
    "snes residual assemble",
    "snes initialize zero matrices",
]


class smartSNESProblem:
    """
    Interface to PETSc SNES solver, defining problem to be solved.

    Args:
        u: The function containing the unknown dofs (is a function from a
            `dolfin.MixedFunctionSpace`)
        Fforms: Nested list of forms for the residual ``F``. Number of
            block rows in the rhs vector is given by the number of items in the outermost list.
        Jforms_all: Nested list of forms for the Jacobian.
            Number of entries in the outermost list should be ``len(Fforms)**2``
            flattened such that ``J[i,j]=sum(Jforms_all[i*len(Fforms)+j])``,
            ``i,j=0,..,len(Fforms)-1``.
            The k-th entry of ``Jforms_all`` is a list of forms that are summed up in a given block.
        active_compartments: List of compartments used in the variational form.
        all_compartments: List of all compartments in model.
        stopwatches: Dictionary of stop-watches (stopwatch_name: stopwatch-class).
            Must include at least `snes jacobian assemble`, `snes residual assemble`
            and `snes initialize zero matrices`
        verbose: If True, output logger info.
    """

    def __init__(
        self,
        u: d.Function,
        Fforms: List[List[d.Form]],
        Jforms_all: List[List[d.Form]],
        active_compartments: List[Compartment],
        all_compartments: List[Compartment],
        stopwatches: Dict[str, Stopwatch],
        verbose: bool,
    ):
        self.u = u
        self.Fforms = Fforms
        self.Jforms_all = Jforms_all
        # for convenience, the mixed function space (model.V)
        self.W = [usub.function_space() for usub in u._functions]
        self.dim = len(self.Fforms)

        assert len(self.Jforms_all) == self.dim**2

        # Extract MPI communicator from one of the underlying meshes
        assert len(self.u._functions) >= 1
        self.comm = self.u._functions[0].function_space().mesh().mpi_comm()
        self.rank = self.comm.rank
        self.size = self.comm.size

        # save sparsity patterns of block matrices
        self.tensors = [[None] * len(Jij_list) for Jij_list in self.Jforms_all]

        # Get local_to_global maps (len=number of owned dofs + ghost dofs) and
        # dofs (len=number of owned dofs)
        self.dofs = [V.dofmap().dofs for V in self.W]
        self.lgmaps = [V.dofmap().tabulate_local_to_global_dofs().astype("int32") for V in self.W]
        self.block_sizes = [max(V.num_sub_spaces(), 1) for V in self.W]

        self.block_indices = [
            (z[::block_size] / block_size).astype("int32")
            for z, block_size in zip(self.lgmaps, self.block_sizes)
        ]
        self.lgmaps_petsc = [
            p.LGMap().create(lgmap, bsize=bsize, comm=self.comm)
            for bsize, lgmap in zip(self.block_sizes, self.lgmaps)
        ]

        self.local_ownership_ranges = [V.dofmap().ownership_range() for V in self.W]
        self.local_sizes = [x[1] - x[0] for x in self.local_ownership_ranges]
        self.global_sizes = [V.dim() for V in self.W]

        # Need sizes because some forms may be empty
        self.is_single_domain = len(self.global_sizes) == 1
        self.active_compartment_names = [c.name for c in active_compartments]
        self.mesh_id_to_name = {c.mesh_id: c.name for c in all_compartments}

        # Should we print assembly info (can get very verbose)
        self.verbose = verbose

        # Timings
        for key in _stopwatch_keys:
            if key not in stopwatches.keys():
                raise ValueError(f"Stopwatch dictionary missing stopwatch {key}")
        self.stopwatches = stopwatches

        # Check for empty forms
        self.empty_forms = []
        for i in range(self.dim):
            for j in range(self.dim):
                ij = i * self.dim + j
                if all(
                    self.Jforms_all[ij][k].function_space(0) is None
                    for k in range(len(self.Jforms_all[ij]))
                ):
                    self.empty_forms.append((i, j))
        if len(self.empty_forms) > 0:
            if self.verbose:
                logger.debug(
                    f"Forms {self.empty_forms} are empty. Skipping assembly.",
                    extra=dict(format_type="data"),
                )

    def init_petsc_matnest(self):
        Jforms = self.Jforms_all
        dim = self.dim
        Jpetsc = []
        for i in range(dim):
            for j in range(dim):
                ij = i * dim + j

                non_empty_forms = 0
                for k in range(len(Jforms[ij])):
                    if Jforms[ij][k].function_space(0) is None:
                        # The only reason this is empty is because the whole form is empty
                        assert len(Jforms[ij]) == 1
                        if self.verbose:
                            logger.debug(
                                f"{self.Jijk_name(i,j,k=None)} is empty",
                                extra=dict(format_type="log"),
                            )
                        continue
                    else:
                        non_empty_forms += 1

                    # initialize the tensor
                    if self.tensors[ij][k] is None:
                        self.tensors[ij][k] = d.PETScMatrix(self.comm)

                    logger.debug(
                        f"cpu {self.rank}: (ijk)={(i,j,k)} "
                        f"({self.local_sizes[i]}, {self.local_sizes[j]}, "
                        f"{self.global_sizes[i]}, {self.global_sizes[j]})",
                        extra=dict(format_type="log"),
                    )

                    d.assemble_mixed(
                        Jforms[ij][k], tensor=self.tensors[ij][k]
                    )  # causes issues when running multiple meshes in parallel

                if non_empty_forms == 0:
                    # If all forms are empty, we don't need to assemble. Initialize to zero matrix
                    if self.verbose:
                        logger.debug(
                            f"{self.Jijk_name(i,j)} is empty - initializing as "
                            f"empty PETSc Matrix with local size {self.local_sizes[i]}, "
                            f"{self.local_sizes[j]} "
                            f"and global size {self.global_sizes[i]}, {self.global_sizes[j]}",
                            extra=dict(format_type="log"),
                        )
                    self.tensors[ij][0] = d.PETScMatrix(self.init_petsc_matrix(i, j, assemble=True))
                    Jpetsc.append(self.tensors[ij][0])
                elif non_empty_forms == 1:
                    Jpetsc.append(self.tensors[ij][0])
                else:
                    # sum the matrices
                    # Because of the nature of these problems, it is a very reasonable
                    # guess that the matrix with the most non-zeros
                    # has a non-zero pattern which is a super-set of the other matrices.
                    nnzs = [M.nnz() for M in self.tensors[ij]]
                    k_max_nnz = nnzs.index(max(nnzs))
                    Jsum = self.d_to_p(self.tensors[ij][k_max_nnz].copy())
                    # Jsum = self.tensors[ij][k_max_nnz].copy()
                    for k in range(len(self.tensors[ij])):
                        if k == k_max_nnz:
                            continue
                        # structure options: SAME_NONZERO_PATTERN, DIFFERENT_NONZERO_PATTERN,
                        # SUBSET_NONZERO_PATTERN, UNKNOWN_NONZERO_PATTERN
                        Jsum.axpy(
                            1,
                            self.d_to_p(self.tensors[ij][k]),
                            structure=Jsum.Structure.SUBSET_NONZERO_PATTERN,
                        )
                    Jpetsc.append(d.PETScMatrix(Jsum))

        if self.is_single_domain:
            # We can't use a nest matrix
            self.Jpetsc_nest = Jpetsc[0].mat()
        else:
            self.Jpetsc_nest = d.PETScNestMatrix(Jpetsc).mat()
            # self.Jpetsc_nest = self.d_to_p(d.PETScNestMatrix(Jpetsc))
        self.Jpetsc_nest.assemble()
        logger.info(f"Jpetsc_nest assembled, size = {self.Jpetsc_nest.size}")

    def d_to_p(self, dolfin_matrix):
        return d.as_backend_type(dolfin_matrix).mat()

    def init_petsc_vecnest(self):
        dim = self.dim
        if self.verbose:
            logger.info("Initializing block residual vector", extra=dict(format_type="assembly"))

        Fpetsc = []
        for j in range(dim):
            Fsum = None
            for k in range(len(self.Fforms[j])):
                if self.Fforms[j][k].function_space(0) is None:
                    if self.verbose:
                        logger.warning(
                            f"{self.Fjk_name(j,k)}] has no function space",
                            extra=dict(format_type="log"),
                        )
                    continue

                tensor = d.PETScVector(self.comm)

                if Fsum is None:
                    Fsum = d.assemble_mixed(self.Fforms[j][k], tensor=tensor)
                else:
                    # Fsum.axpy(1, d.assemble_mixed(self.Fforms[j][k], tensor=tensor).vec(),
                    # structure=Fsum.Structure.DIFFERENT_NONZERO_PATTERN)
                    Fsum += d.assemble_mixed(self.Fforms[j][k], tensor=tensor)

            if Fsum is None:
                if self.verbose:
                    logger.debug(
                        f"{self.Fjk_name(j)} is empty - initializing as empty PETSc "
                        f"Vector with local size {self.local_sizes[j]} "
                        f"and global size {self.global_sizes[j]}",
                        extra=dict(format_type="log"),
                    )
                Fsum = d.PETScVector(self.init_petsc_vector(j, assemble=True))

            Fpetsc.append(Fsum.vec())

        if self.is_single_domain:
            # We can't use a nest vector
            self.Fpetsc_nest = d.PETScVector(Fpetsc[0]).vec()
        else:
            self.Fpetsc_nest = p.Vec().createNest(Fpetsc, comm=self.comm)
        self.Fpetsc_nest.assemble()

    def assemble_Jnest(self, Jnest):
        """Assemble Jacobian nest matrix.

        Args:
            Jnest : petsc4py.Mat
                PETSc nest matrix representing the Jacobian


        """
        if self.verbose:
            logger.debug("Assembling block Jacobian", extra=dict(format_type="assembly"))
        self.stopwatches["snes jacobian assemble"].start()
        dim = self.dim

        Jform = self.Jforms_all

        # Get the petsc sub matrices, convert to dolfin wrapper, assemble forms using
        # dolfin wrapper as tensor
        for i in range(dim):
            for j in range(dim):
                if (i, j) in self.empty_forms:
                    continue
                ij = i * dim + j
                num_subforms = len(Jform[ij])

                # Extract petsc submatrix
                if self.is_single_domain:
                    Jij_petsc = Jnest
                else:
                    Jij_petsc = Jnest.getNestSubMatrix(i, j)
                Jij_petsc.zeroEntries()  # this maintains sparse (non-zeros) structure

                if self.verbose:
                    logger.debug(
                        f"Assembling {self.Jijk_name(i,j)}:",
                        extra=dict(format_type="assembly_sub"),
                    )

                Jmats = []
                # Jijk == dFi/duj(Omega_k)
                for k in range(num_subforms):
                    # Check for empty form
                    if Jform[ij][k].function_space(0) is None:
                        if self.verbose:
                            logger.debug(
                                f"{self.Jijk_name(i,j,k)} is empty. Skipping assembly.",
                                extra=dict(format_type="data"),
                            )
                        continue

                    # if we have the sparsity pattern re-use it, if not save it for next time
                    # single domain can't re-use the tensor for some reason
                    if self.tensors[ij][k] is None and not self.is_single_domain:
                        raise AssertionError("I dont think this should happpen")
                    elif self.is_single_domain:
                        self.tensors[ij][k] = d.PETScMatrix(self.comm)
                    else:
                        if self.verbose:
                            logger.debug(
                                f"Reusing tensor for {self.Jijk_name(i,j,k)}",
                                extra=dict(format_type="data"),
                            )
                    # Assemble and append to the list of subforms
                    Jmats.append(d.assemble_mixed(Jform[ij][k], tensor=self.tensors[ij][k]))
                    # Print some useful info on assembled Jijk
                    self.print_Jijk_info(i, j, k, tensor=self.tensors[ij][k].mat())

                # Sum the assembled forms
                for Jmat in Jmats:
                    # structure options: SAME_NONZERO_PATTERN, DIFFERENT_NONZERO_PATTERN,
                    # SUBSET_NONZERO_PATTERN, UNKNOWN_NONZERO_PATTERN
                    Jij_petsc.axpy(
                        1,
                        self.d_to_p(Jmat),
                        structure=Jij_petsc.Structure.SUBSET_NONZERO_PATTERN,
                    )

                self.print_Jijk_info(i, j, k=None, tensor=Jij_petsc)

        Jnest.assemble()

        self.stopwatches["snes jacobian assemble"].pause()

    def assemble_Fnest(self, Fnest):
        """
        Assemble residual nest vector

        Arguments:
            Fnest : petsc4py.Vec
                PETSc nest vector representing the residual
        """
        dim = self.dim
        if self.verbose:
            logger.debug("Assembling block residual vector", extra=dict(format_type="assembly"))
        self.stopwatches["snes residual assemble"].start()

        if self.is_single_domain:
            Fj_petsc = [Fnest]
        else:
            Fj_petsc = Fnest.getNestSubVecs()
        Fvecs = []

        for j in range(dim):
            Fvecs.append([])
            for k in range(len(self.Fforms[j])):
                # , tensor=d.PETScVector(Fvecs[idx]))
                Fvecs[j].append(d.as_backend_type(d.assemble_mixed(self.Fforms[j][k])))
            # TODO: could probably speed this up by not using axpy if there is only one subform
            # sum the vectors
            Fj_petsc[j].zeroEntries()
            for k in range(len(self.Fforms[j])):
                Fj_petsc[j].axpy(1, Fvecs[j][k].vec())

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

    def init_petsc_matrix(self, i, j, nnz_guess=None, set_lgmap=False, assemble=False):
        """
        Initialize a PETSc matrix with appropriate structure

        Args:
            i,j : indices of the block
            nnz_guess : number of non-zeros (per row) to guess for the matrix
            assemble : whether to assemble the matrix or not (Boolean)
        """
        self.stopwatches["snes initialize zero matrices"].start()

        M = p.Mat().create(comm=self.comm)
        # ((local_nrows, global_nrows), (local_ncols, global_ncols))
        M.setSizes(
            (
                (self.local_sizes[i], self.global_sizes[i]),
                (self.local_sizes[j], self.global_sizes[j]),
            )
        )
        # M.setBlockSizes(self.block_sizes[i], self.block_sizes[j])
        # M.setBlockSizes(1,1) # seems to be ok with block size 1?
        M.setType("aij")  # "baij"

        if nnz_guess is not None:
            M.setPreallocationNNZ([nnz_guess, nnz_guess])

        # just slow down quietly if preallocation is insufficient
        # M.setOption(p.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

        M.setUp()

        if set_lgmap:
            M.setLGMap(self.lgmaps_petsc[i], self.lgmaps_petsc[j])

        if assemble:
            M.assemble()
        self.stopwatches["snes initialize zero matrices"].pause()

        return M

    def init_petsc_vector(self, j, assemble=False):
        """Initialize a dolfin wrapped PETSc vector with appropriate structure

        Args:
            j : index
            assemble : whether to assemble the vector or not (Boolean)
        """
        V = p.Vec().create(comm=self.comm)
        V.setSizes((self.local_sizes[j], self.global_sizes[j]))
        V.setUp()
        # V.setLGMap(self.lgmaps_petsc[j])

        if assemble:
            V.assemble()
        return V

    def Jijk_name(self, i: int, j: int, k: Optional[int] = None):
        """Get a string representation of an entry of the Jacobian.

        Args:
            i: Row index
            j: Column index
            k: If the Jacobian entry is a sum of forms, get the name
                of the domain in the k'th entry.
        """
        if k is None:
            return (
                f"J{i}{j} = dF[{self.active_compartment_names[i]}]"
                f"/du[{self.active_compartment_names[j]}]"
            )
        else:
            ij = i * self.dim + j
            if (num_entries := len(self.Jforms_all[ij])) <= k:
                raise RuntimeError(f"J[{i},{j}] only consists of {num_entries} componets")

            domain_name = self.mesh_id_to_name[self.Jforms_all[ij][k].function_space(0).mesh().id()]
            return (
                f"J{i}{j}{k} = dF[{self.active_compartment_names[i]}]"
                f"/du[{self.active_compartment_names[j]}] (domain={domain_name})"
            )

    def Fjk_name(self, j: int, k: Optional[int] = None):
        """
        Get a string representation of an entry of the residual.

        Args:
            j: Block index
            k: If the residual entry is a sum of forms, get the
                name of the domain in the k'th entry.
        """

        if k is None:
            return f"F{j} = F[{self.active_compartment_names[j]}]"
        else:
            if (num_entries := len(self.Fforms[j])) <= k:
                raise RuntimeError(f"F[{j}] only consists of {num_entries} componets")
            domain_name = self.mesh_id_to_name[self.Fforms[j][k].function_space(0).mesh().id()]
            return f"F{j} = F[{self.active_compartment_names[j]}] (domain={domain_name})"

    def print_Jijk_info(self, i, j, k=None, tensor=None):
        "Print information on Jacobian nest matrix"
        if not self.verbose:
            return
        if tensor is None:
            return
        # Print some useful info on Jijk
        info = tensor.getInfo()
        # , block_size={int(info['block_size'])}
        info_str = (
            f"size={str(tensor.size)[1:-1]: <18}, nnz={int(info['nz_allocated']): <8}, "
            f"memory[MB]={int(1e-6*info['memory']): <6}, "
            f"assemblies={int(info['assemblies']): <4}, "
            f"mallocs={int(info['mallocs']): <4}\n"
        )
        if k is None:
            logger.debug(
                f"Assembled form {self.Jijk_name(i,j,k)}:\n{info_str}",
                extra=dict(format_type="data"),
            )
        else:
            logger.debug(
                f"Assembled subform {self.Jijk_name(i,j,k)}:\n{info_str}",
                extra=dict(format_type="data"),
            )
        if info["nz_unneeded"] > 0:
            logger.warning(
                f"WARNING: {info['nz_unneeded']} nonzero entries are unneeded",
                extra=dict(format_type="warning"),
            )

    def get_csr_matrix(self, i, j):
        """This is a matrix that can be used to visualize
        the sparsity pattern using :code:`plt.spy()`"""
        if self.is_single_domain:
            M = self.Jpetsc_nest
        else:
            M = self.Jpetsc_nest.getNestSubMatrix(i, j)
        from scipy.sparse import csr_matrix

        return csr_matrix(M.getValuesCSR()[::-1], shape=M.size)
