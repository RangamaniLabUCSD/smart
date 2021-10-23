#=======================================================
# test2_poisson.py
# Solving the Poisson equation using dolfinx
#=======================================================
from dolfinx import (Function, FunctionSpace, RectangleMesh, fem, plot)
from ufl import ds, dx, grad, inner
