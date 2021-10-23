import dolfin as d

n = 10
mesh_v = d.UnitCubeMesh(n,n,n)
d.MeshView()