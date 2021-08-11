stubs
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.org/justinlaughlin/stubs.png)](https://travis-ci.org/justinlaughlin/stubs)
[![AppVeyor Build status](https://ci.appveyor.com/api/projects/status/REPLACE_WITH_APPVEYOR_LINK/branch/master?svg=true)](https://ci.appveyor.com/project/justinlaughlin/stubs/branch/master)
[![codecov](https://codecov.io/gh/justinlaughlin/stubs/branch/master/graph/badge.svg)](https://codecov.io/gh/justinlaughlin/stubs/branch/master)

Fenics stuff

### Copyright

Copyright (c) 2019, Justin Laughlin


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.


#### Nomenclature

Because STUBS methods are viable in both 3 dimensional and 2 dimensional geometries we use the following nomenclature to
define various functions in the code.

Cell            : The element of the highest geometric dimension (e.g. "cell" refers to a tetrahedra in 3d, but a triangle in 2d).
Facet           : The element of dimenion n-1 if n is the highest geometric dimension.
Volume mesh     : A set of elements of the highest geometric dimension.
Surface mesh    : A set of elements of dimension n-1 if n is the highest geometric dimension.

#### Standard example meshes



#### Notes on how FEniCS functions treat meshes



