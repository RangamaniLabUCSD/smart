##########################
Mathematics
##########################

Mathematics related to stubs.

.. _Multi-Dimensional Reaction-Diffusion Equations:

**************************************************
Multi-Dimensional Reaction-Diffusion Equations
**************************************************

Volumetric partial differential equations

.. math::
   \partial_t u^{(a)}_{i} &= \nabla \cdot (D^{(a)}_{i} \nabla u^{(a)}_{i}) + f^{(a)}_{i}(u^{(a)}) ~~\text{in}~~ \Omega^{(a)}\\
   D^{(a)}_{i} (\nabla u^{(a)}_{i} \cdot n) &= r^{(abc)}_{i}(u^{(a)}, u^{(b)}, v^{(abc)}) ~~\text{on}~~ \Gamma^{(abc)}

Surface partial differential equations

.. math::
   \partial_t v^{(abc)}_{i} &= \nabla_S \cdot (D^{(abc)}_{i} \nabla_S v^{(abc)}_{i}) + g^{(abc)}_{i}(u^{(a)}, u^{(b)}, v^{(abc)}) ~~\text{on}~~ \Gamma^{(abc)} \\
   D^{(abc)}_{i} (\nabla_S v^{(abc)}_i \cdot n) &= 0 ~~\text{on}~~ \partial\Gamma^{(abc)}
