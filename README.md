[![Test fenics_smart](https://github.com/RangamaniLabUCSD/smart/actions/workflows/test_fenics_smart.yml/badge.svg)](https://github.com/RangamaniLabUCSD/smart/actions/workflows/test_fenics_smart.yml)
[![PyPI](https://img.shields.io/pypi/v/fenics-smart)](https://pypi.org/project/fenics-smart/)
[![Deploy static content to Pages](https://github.com/RangamaniLabUCSD/smart/actions/workflows/build_docs.yml/badge.svg)](https://github.com/RangamaniLabUCSD/smart/actions/workflows/build_docs.yml)
[![pre-commit](https://github.com/RangamaniLabUCSD/smart/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/RangamaniLabUCSD/smart/actions/workflows/pre-commit.yml)
# Spatial Modeling Algorithms for Reaction-Transport [systems|models|equations]

## Statement of Need

*Spatial Modeling Algorithms for Reactions and Transport* (SMART) is a high-performance finite-element-based simulation package for model specification and numerical simulation of spatially-varying reaction-transport processes,
especially tailored to modeling such systems within biological cells.
SMART is based on the [FEniCS finite element library](https://fenicsproject.org/), provides a symbolic representation
framework for specifying reaction pathways, and supports large and irregular cell geometries in 2D and 3D.

- Documentation: https://rangamanilabucsd.github.io/smart
- Source code: https://github.com/RangamaniLabUCSD/smart


## Installation

### Using docker (recommended)
The simplest way to use `fenics-smart` is to use the provided docker image. You can get this image by pulling it from the github registry
```
docker pull ghcr.io/rangamanilabucsd/smart:latest
```
It is also possible to pull a specific version by changing the tag, e.g
```
docker pull ghcr.io/rangamanilabucsd/smart:v2.0.1
```
will use version 2.0.1.

In order to start a container you can use the [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) command. For example the command
```
docker run --rm -v $(pwd):/home/shared -w /home/shared -ti ghcr.io/rangamanilabucsd/smart:latest
```
will run the latest version and share your current working directory with the container.

### Using pip
`fenics-smart` is also available on [pypi](https://pypi.org/project/fenics-smart/) and can be installed with
```
python3 -m pip install fenics-smart
```
However this requires FEniCS version 2019.2.0 or later to already be installed. Currently, FEniCS version 2019.2.0 needs to be built [from source](https://bitbucket.org/fenics-project/dolfin/src/master/) or use some of the [pre-built docker images](https://github.com/orgs/scientificcomputing/packages?repo_name=packages)

## Example usage
The SMART repository contains a number of examples in the `examples` directory which also run as continuous integration tests (see "Automated Tests" below):
* [Example 1](https://rangamanilabucsd.github.io/smart/examples/example1/example1.html): Formation of Turing patterns in 2D reaction-diffusion (rectangular domain)
* [Example 2](https://rangamanilabucsd.github.io/smart/examples/example2/example2.html): Simple cell signaling model in 2D (ellipse)
* [Example 3](https://rangamanilabucsd.github.io/smart/examples/example3/example3.html): Model of protein phosphorylation and diffusion in 3D (sphere)
* [Example 4](https://rangamanilabucsd.github.io/smart/examples/example4/example4.html): Model of second messenger reaction-diffusion in 3D (ellipsoid-in-an-ellipsoid)
* [Example 5](https://rangamanilabucsd.github.io/smart/examples/example5/example5.html): Simple cell signaling model in 3D (cube-in-a-cube)
* [Example 6](https://rangamanilabucsd.github.io/smart/examples/example6/example6.html): Model of calcium dynamics in a neuron (sphere-in-a-sphere)

## Functionality documentation
SMART is equipped to handle:
* Reaction-diffusion with any number of species, reactions, and compartments.
* 3D-2D problems or 2D-1D problems; that is, you can solve a problem with many 3d sub-volumes coupled to many 2d sub-surfaces, or a problem with many 2d "sub-volumes" coupled to many 1d "sub-surfaces"
* Conversion of units at run-time via [Pint](https://pint.readthedocs.io/en/stable/) so that models can be specified in whatever units are most natural/convenient to the user.
* Specification of a time-dependent function either algebraically or from data (SMART will numerically integrate the data points at each time-step).
* Customized reaction equations (e.g. irreversible Hill equation).

The general form of the mixed-dimensional partial differential equations (PDEs) solved by SMART, along with mathematical details of the numerical implementation, are documented [here](https://rangamanilabucsd.github.io/smart/docs/math.html).

Our API documentation can be accessed [here](https://rangamanilabucsd.github.io/smart/docs/api.html).

## Automated tests
Upon pushing new code to the SMART repository, a number of tests run:
* pre-commit tests
* unit tests (can be found in `tests` folder): test initialization of compartment, species, and parameter objects
* Examples 1-6: All 6 examples are run when building the docs. These serve as Contiuous Integration (CI) tests; within each run, there is a regression test comparing the output values from the simulation with values obtained from a previous build of SMART. Outputs from examples 2 and 3 are also compared to analytical solutions to demonstrate the accuracy of SMART simulations.
* Example 3 with MPI: Example 3 is run using MPI to run differently sized meshes in parallel (each process is assigned a single mesh).

## Contributing guidelines

Detailed contributing guidelines are given [here](https://rangamanilabucsd.github.io/smart/CONTRIBUTING.html).

### Dependencies
* SMART uses [FEniCS](https://fenicsproject.org/) to assemble finite element matrices from the variational form
* SMART uses [PETSc4py] to solve the resultant linear algebra systems.
* SMART uses [pandas](https://pandas.pydata.org/) as an intermediate data structure to help organize and process models.
* SMART uses [Pint](https://pint.readthedocs.io/en/stable/) for unit tracking and conversions.
* SMART uses [matplotlib](https://matplotlib.org/) to generate plots in examples
* SMART uses [sympy](https://www.sympy.org/) to allow users to input custom reactions and also to determine the appopriate solution techniques (e.g. testing for non-linearities).
* SMART uses [numpy](https://numpy.org/) and [scipy](https://www.scipy.org/) for general array manipulations and basic calculations.
* SMART uses [tabulate](https://pypi.org/project/tabulate/) to make ASCII tables.
* SMART uses [termcolor](https://pypi.org/project/termcolor/) for colored terminal output.

## License
LGPL-3.0

## SMART development team
* [Justin Laughlin](https://github.com/justinlaughlin) - original author of the repository
* [Christopher Lee](https://github.com/ctlee)
* [Emmet Francis](https://github.com/emmetfrancis)
* [Jorgen Dokken](https://github.com/jorgensd)
* [Henrik Finsberg](https://github.com/finsberg)

Previous contributors:
* [Yuan Gao](https://github.com/Rabona17)
* [William Xu](https://github.com/willxu1234)
