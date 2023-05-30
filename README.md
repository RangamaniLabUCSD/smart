[![Test fenics_smart](https://github.com/RangamaniLabUCSD/smart/actions/workflows/test_fenics_smart.yml/badge.svg)](https://github.com/RangamaniLabUCSD/smart/actions/workflows/test_fenics_smart.yml)
[![PyPI](https://img.shields.io/pypi/v/fenics-smart)](https://pypi.org/project/fenics-smart/)
[![Deploy static content to Pages](https://github.com/RangamaniLabUCSD/smart/actions/workflows/build_docs.yml/badge.svg)](https://github.com/RangamaniLabUCSD/smart/actions/workflows/build_docs.yml)
[![pre-commit](https://github.com/RangamaniLabUCSD/smart/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/RangamaniLabUCSD/smart/actions/workflows/pre-commit.yml)
# Spatial Modelling Algorithms for Reaction-Transport [systems|models|equations]

## Statement of Need

SMART is a biophysical simulation library that provides a level of abstraction to models, making it easier for users to develop, share, and simulate their mathematical models.
SMART is highly suited for building systems biology models and simulating them as deterministic partial differential equations (PDEs)` in realistic geometries using the Finite Element Method (FEM).
Systems biology models are converted by SMART into the appropriate systems of reaction-diffusion PDEs with proper boundary conditions.
[FEniCS](https://fenicsproject.org/) is a core dependency of SMART which handles the assembly of finite element matrices as well as solving the resultant linear algebra systems.

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
The SMART repository contains a number of examples in the `examples` directory:
* Example 1: Single species reaction-diffusion in a sphere
* Example 2: Single species reaction-diffusion with two volumetric compartments (sphere within a sphere) and two surface compartments
* Example 3: Multispecies reaction-diffusion across two volumetric compartments and two surface compartments - cube in a cube
* Example 4: Simplified model of calcium dynamics in a neuron - sphere in a sphere

## Functionality documentation
SMART is equipped to handle:
* Reaction-diffusion with any number of species, reactions, and compartments.
* 3D-2D problems or 2D-1D problems; that is, you can solve a problem with many 3d sub-volumes coupled to many 2d sub-surfaces, or a problem with many 2d "sub-volumes" coupled to many 1d "sub-surfaces"
* Conversion of units at run-time via [Pint](https://pint.readthedocs.io/en/stable/) so that models can be specified in whatever units are most natural/convenient to the user.
* Specification of a time-dependent function either algebraically or from data (SMART will numerically integrate the data points at each time-step).
* Customized reaction equations (e.g. irreversible Hill equation).

The general form of the mixed-dimensional partial differential equations (PDEs) solved by SMART, along with mathematical details of the numerical implementation, are documented here.

Our API documentation can be accessed here.

## Automated tests
Upon pushing new code to the SMART repository, a number of tests run:
* pre-commit tests
* unit tests
* Examples

## Contributing guidelines

Detailed contributing guidelines are given here.

### Dependencies
* SMART uses [FEniCS](https://fenicsproject.org/) to assemble finite element matrices from the variational form
* SMART uses [PETSc4py] to solve the resultant linear algebra systems.
* SMART uses [pandas](https://pandas.pydata.org/) as an intermediate data structure to help organize and process models.
* SMART uses [Pint](https://pint.readthedocs.io/en/stable/) for unit tracking and conversions.
* SMART uses [sympy](https://www.sympy.org/) to allow users to input custom reactions and also to determine the appopriate solution techniques (e.g. testing for non-linearities).
* SMART uses [numpy](https://numpy.org/) and [scipy](https://www.scipy.org/) for general array manipulations and basic calculations.
* SMART uses [tabulate](https://pypi.org/project/tabulate/) to make pretty ASCII tables.
* SMART uses [termcolor](https://pypi.org/project/termcolor/) for pretty terminal output so that simulations are more satisfying to watch.

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
