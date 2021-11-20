###############################################
Setup Tool for Unified Biophysical Simulations
###############################################

STUBS is a biophysical simulation library that provides a level of abstraction to models, making it easier for users to develop, share, and simulate their mathematical models. 
STUBS is highly suited for building systems biology models and simulating them as deterministic partial differential equations `[PDEs]` in realistic geometries using the Finite Element Method `[FEM]` - the integration of additional physics such as electro-diffusion or stochasticity may come in future updates. 
Systems biology models are converted by STUBS into the appropriate systems of reaction-diffusion PDEs with proper boundary conditions. 
`FEniCS <https://fenicsproject.org/>`_ is a core dependency of STUBS which handles the assembly of finite element matrices as well as solving the resultant linear algebra systems. 

.. toctree::
   :maxdepth: 1
   :caption: Getting Started:

   install
   math
   faq

.. toctree::
   :maxdepth: 1
   :caption: API Documentation:

   pystubs


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
