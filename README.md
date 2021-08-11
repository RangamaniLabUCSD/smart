Setup Tool for Unified Biophysical Simulations
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.org/justinlaughlin/stubs.png)](https://travis-ci.org/justinlaughlin/stubs)
[![AppVeyor Build status](https://ci.appveyor.com/api/projects/status/REPLACE_WITH_APPVEYOR_LINK/branch/master?svg=true)](https://ci.appveyor.com/project/justinlaughlin/stubs/branch/master)
[![codecov](https://codecov.io/gh/justinlaughlin/stubs/branch/master/graph/badge.svg)](https://codecov.io/gh/justinlaughlin/stubs/branch/master)

STUBS is a biophysical simulation library that provides a level of abstraction to models, making it easier for users to develop, share, and simulate their mathematical models. 
STUBS is highly suited for building systems biology models and simulating them in realistic geometries. 
Systems biology models are converted by STUBS into the appropriate systems of reaction-diffusion  partial differential equations `[PDEs]` with proper boundary conditions. 
FEniCS (https://fenicsproject.org/) is a core dependency of STUBS which handles the assembly of finite element `[FEM]` matrices as well as solving the system. 
Coupled problems (e.g. reaction-diffusion in a volume coupled to reaction-diffusion on a surface) are decoupled using the Picard (fixed-point) method while non-linearities within the systems themselves are handled by FEniCS' implementation of Newton's method. 


STUBS is highly suited for building systems biology models and simulating them in realistic geometries (STUBS will construct the appropriate reaction-diffusion partial differential equations `[PDEs]` with proper boundary conditions). 
FEniCS (https://fenicsproject.org/) is a core dependency of STUBS - Systems biology models are converted into appropriate reaction-diffusion partial differential equations `[PDEs]` with proper boundary conditions the backend for any PDE simulations. 
STUBS is equipped to handle:

* Any number of species, reactions, and compartments
* Reaction-diffusion in a volume


## External libraries bundled/downloaded with/by STUBS
* STUBS uses [FEniCS] (https://fenicsproject.org/)


## Nomenclature

Because STUBS methods are viable in both 3 dimensional and 2 dimensional geometries we use the following nomenclature to define various functions in the code.

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

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
