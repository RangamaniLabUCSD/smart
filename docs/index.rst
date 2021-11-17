###########################################
Setup Tool for Unified Biophysical Simulations
###########################################

STUBS is a biophysical simulation library that provides a level of abstraction to models, making it easier for users to develop, share, and simulate their mathematical models. 
STUBS is highly suited for building systems biology models and simulating them as deterministic partial differential equations `[PDEs]` in realistic geometries using the Finite Element Method `[FEM]` - the integration of additional physics such as electro-diffusion or stochasticity may come in future updates. 
Systems biology models are converted by STUBS into the appropriate systems of reaction-diffusion PDEs with proper boundary conditions. 
`FEniCS <https://fenicsproject.org/>`_ is a core dependency of STUBS which handles the assembly of finite element matrices as well as solving the resultant linear algebra systems. 


## Installation

### !!! IMPORTANT !!!
Although FEniCS is a core dependency, because it has many different versions (2019.1, development, FEniCSx, etc.), is quite large, and is complicated to build, it is not packaged with STUBS by default. The recommended way to use STUBS is to create a container from one of the official FEniCS docker images and to pip install STUBS from within the container.

```bash
# create a container from the official fenics docker image
jgl:~$ docker run -ti --init quay.io/fenicsproject/dev
# pip install stubs from within the container
fenics:~$ pip install stubs
```

### Dependencies
* STUBS uses `FEniCS <https://fenicsproject.org/>`_ to assemble finite element matrices as well as solve the resultant linear algebra systems.
* STUBS uses `pandas <https://pandas.pydata.org/>`_ as an intermediate data structure to help organize and process models.
* STUBS uses `Pint <https://pint.readthedocs.io/en/stable/>`_ for unit tracking and conversions.
* STUBS uses `matplotlib <https://matplotlib.org/>`_ to automatically generate plots of min/mean/max (integrated over space) concentrations over time, as well as plots showing solver convergence.
* STUBS uses `sympy <https://www.sympy.org/>`_ to allow users to input custom reactions and also to determine the appopriate solution techniques (e.g. testing for non-linearities).
* STUBS uses `numpy <https://numpy.org/>`_ and `scipy <https://www.scipy.org/>`_ for general array manipulations and basic calculations.
* STUBS uses `tabulate <https://pypi.org/project/tabulate/>`_ to make pretty ASCII tables.
* STUBS uses `termcolor <https://pypi.org/project/termcolor/>`_ for pretty terminal output so that simulations are more satisfying to watch.


Welcome to stubs's documentation!
=================================
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
