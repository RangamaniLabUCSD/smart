# Software implementation

The overall SMART workflow is summarized in the diagram below. In short, the model is initialized by defining SMART containers for the species, compartments, parameters, and reactions, as well as loading in a mesh describing the geometry. With this information and the solver specifications, the variational problem and solver are initialized and then the solution is computed at each time step.

![SMART flow chart](https://github.com/RangamaniLabUCSD/smart/assets/99422170/50d7f57c-1250-4078-a8a0-a74194c45852)

The SMART abstractions and algorithms are implemented via the open source and generally available FEniCS finite element software package (2022-version) {cite}`alnaes2015fenics`. FEniCS supports high-level specification of variational forms via the Unified Form Language (UFL) {cite}`alnaes2014unified`, symbolic differentiation of variational forms e.g.~for derivation for Jacobians, automated assembly of general and nonlinear forms over finite element meshes, and high-performance linear algebra and nonlinear solvers via e.g. PETSc {cite}`balay1998petsc`.


## References

```{bibliography}
:filter: docname in docnames
```
