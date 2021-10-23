#=======================================================
# test1_subdomain_extraction.py
# Extracting subvolumes and subsurfaces from a mesh 
# using dolfinx
# Some good files to look at
# * test_xdmf_meshtags.py
# * demo_gmsh.py
#=======================================================
import dolfinx
import numpy as np
import ufl

from dolfinx import (RectangleMesh, fem, plot)
from dolfinx.cpp.mesh import CellType
from dolfinx.mesh import MeshTags, locate_entities, locate_entities_boundary

from mpi4py import MPI
from petsc4py import PETSc
from ufl import MeshView

mesh = RectangleMesh(
    MPI.COMM_WORLD,
    [np.array([0,0,0]), np.array([1,1,0])], [4,4],
    CellType.triangle, dolfinx.cpp.mesh.GhostMode.none)

mesh_cube = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, 4, 4, 4, CellType.tetrahedron)


#meshview = MeshView(mesh.ufl_domain(), 2)

cells_0 = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, lambda x: x[0]<=0.5)
cells_1 = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, lambda x: x[0]>=0.5)


from mpi4py import MPI
import numpy as np
import dolfinx
from dolfinx import (RectangleMesh, FunctionSpace)
from dolfinx.cpp.mesh import CellType

mesh = RectangleMesh(
    MPI.COMM_WORLD,
    [np.array([0,0,0]), np.array([1,1,0])], [4,4],
    CellType.triangle, dolfinx.cpp.mesh.GhostMode.none)
V  = FunctionSpace(mesh, ("P", 1)) # Linear Lagrange mesh
V2 = FunctionSpace(mesh, ("P", 2)) # Quadratic Lagrange mesh


