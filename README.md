Spatial Modelling Algorithms for Reaction-Transport [systems|models|equations]
==============================
[//]: # (Badges)
[![PyPI](https://img.shields.io/pypi/v/fenics-smart)](https://pypi.org/project/fenics-smart/)


SMART is a biophysical simulation library that provides a level of abstraction to models, making it easier for users to develop, share, and simulate their mathematical models.
SMART is highly suited for building systems biology models and simulating them as deterministic partial differential equations `[PDEs]` in realistic geometries using the Finite Element Method `[FEM]` - the integration of additional physics such as electro-diffusion or stochasticity may come in future updates.
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


### Dependencies
* SMART uses [FEniCS](https://fenicsproject.org/) to assemble finite element matrices as well as solve the resultant linear algebra systems.
* SMART uses [pandas](https://pandas.pydata.org/) as an intermediate data structure to help organize and process models.
* SMART uses [Pint](https://pint.readthedocs.io/en/stable/) for unit tracking and conversions.
* SMART uses [matplotlib](https://matplotlib.org/) to automatically generate plots of min/mean/max (integrated over space) concentrations over time, as well as plots showing solver convergence.
* SMART uses [sympy](https://www.sympy.org/) to allow users to input custom reactions and also to determine the appopriate solution techniques (e.g. testing for non-linearities).
* SMART uses [numpy](https://numpy.org/) and [scipy](https://www.scipy.org/) for general array manipulations and basic calculations.
* SMART uses [tabulate](https://pypi.org/project/tabulate/) to make pretty ASCII tables.
* SMART uses [termcolor](https://pypi.org/project/termcolor/) for pretty terminal output so that simulations are more satisfying to watch.

## Functionality
SMART is equipped to handle:
* Reaction-diffusion with any number of species, reactions, and compartments.
* Reaction-diffusion with boundary conditions between coupled sub-volumes and sub-surfaces (defined by marker values in the .xml file).
* Reaction-diffusion in non-manifold meshes (experimental).
* Conversion of units at run-time via [Pint](https://pint.readthedocs.io/en/stable/) so that models can be specified in whatever units are most natural/convenient to the user.
* Specification of a time-dependent function either algebraically or from data (SMART will numerically integrate the data points at each time-step).
* Customized reaction equations (e.g. irreversible Hill equation).

SMART does not handle (it is possible to implement these features but would require a lot of work - contact author if interested):
* Problems with coupled-physics spanning more than two dimensions. For example you may solve a problem with many 3d sub-volumes coupled to many 2d sub-surfaces, or a problem with many 2d "sub-volumes" coupled to many 1d "sub-surfaces" but a problem with 3d physics can't be coupled to a problem with 1d physics.
* Sub-volumes embedded within sub-volumes (i.e. from any point inside the interior sub-volume, one must traverse two surfaces to get to the exterior of the full mesh)

## Nomenclature
Because SMART methods are viable in both 3 dimensional and 2 dimensional geometries we use the following nomenclature to define various functions in the code.

Cell            : The element of the highest geometric dimension (e.g. "cell" refers to a tetrahedra in 3d, but a triangle in 2d).
Facet           : The element of dimenion n-1 if n is the highest geometric dimension.
Volume mesh     : A set of elements of the highest geometric dimension.
Surface mesh    : A set of elements of dimension n-1 if n is the highest geometric dimension.

"Cell" and "Volume" are used interchangeably (e.g. a volume mesh is a collection of cells). "Facet" and "Surface" are used interchangeably.

## License
STUBS is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

STUBS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with STUBS. If not, see <http://www.gnu.org/licenses/>.

## Acknowledgements

Thanks to [Christopher Lee](https://github.com/ctlee), [Yuan Gao](https://github.com/Rabona17), and [William Xu](https://github.com/willxu1234) for their valuable input and contributions to STUBS.
