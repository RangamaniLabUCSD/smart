# testing projection error with mixed dimensional code
# mwe
from locale import ABDAY_1
import dolfin as d
import numpy as np
import termplotlib as tpl
n = 10
m = d.UnitCubeMesh(n,n,n)

# Make the submesh
mf2 = d.MeshFunction("size_t", m, 2, 0)
z_slices = [0, 0.5]
#z_slices = [0, 1.0]
for z_slice in z_slices:
    for f in d.facets(m):
        mf2[f] += (z_slice-d.DOLFIN_EPS <= f.midpoint().z() <= z_slice+d.DOLFIN_EPS)*1

m_ = d.MeshView.create(mf2, 1)
print(m_.num_cells())

for c in d.cells(m):
    assert np.isclose(c.facet_area(0), 0.005)

dx = d.Measure('dx', domain=m)
dx_ = d.Measure('dx', domain=m_)

# function spaces, test functions
V = d.FunctionSpace(m, "CG", 1)
v = d.TestFunction(V)
V_ = d.FunctionSpace(m_, "CG", 1)
W = d.MixedFunctionSpace(V, V_)



# assemble
f = 1*v*dx_
fv0 = d.assemble(f)
fv = d.assemble_mixed(f)
fsum = sum(fv)
fother = d.assemble_mixed(1*dx_)
print(fv.size())
print(fsum)
print(fother)
analytic_result = m_.num_cells() * 0.005
print(analytic_result)
# =============================
# (03/09/22): fv0 and fv have different results??


fig = tpl.figure()
fig.plot(range(fv.size()), fv)
fig.show()

fig.plot(range(fv0.size()), fv0)
fig.show()



ut0, ut1 = d.TrialFunctions(W)
v0, v1   = d.TestFunctions(W)
usolve = d.Function(W)
#form = d.inner(d.grad(ut0), d.grad(v0))*dx - d.Constant(1)*v0*dx + d.Constant(2)*v0*dx_ + (ut1-3)*v1*dx_
# a = d.inner(d.grad(ut0), d.grad(v0))*dx + ut1*v1*dx_ + ut0*v0*dx
# L = d.Constant(1)*v0*dx + d.Constant(3)*v1*dx_
a0 = d.inner(d.grad(ut0), d.grad(v0))*dx + (ut0-ut1)*v0*dx_
d.assemble_mixed(ut0*v0*dx_)
L0 = 0#d.Constant(1)*v0*dx 
a1 = ut1*v1*dx_
L1 = 3*v1*dx_
#d.solve(a==L,usolve)
d.solve(a0+a1==L0+L1,usolve)



a = u*v*dx_
L = 1*v*dx_

u = d.Function(V)
a = u*v*dx_




# segfaults
# for c in d.cells(m_):
#     assert np.isclose(c.facet_area(0), 0.005)



# vector function space
V = d.VectorFunctionSpace(m, "CG", 1, dim=3)
u = d.Function(V)
u.vector()[:] += np.random.rand(u.vector().size())
u0 = d.split(u)[0]
v = d.TestFunctions(V)
v0 = v[0]

sum(d.assemble_mixed(u0*v0*dx_))
d.assemble_mixed(u0*dx_)



V_ = d.FunctionSpace(m_, "CG", 1)
u2 = d.interpolate(d.Expression('pow(x[0]+x[1]+x[2],2)', degree=1), V_)
sum(d.assemble_mixed(u*u2*v*dx_))
d.assemble_mixed(u*u2*dx_)

sum(d.assemble_mixed(u3*v*dx_))
d.assemble_mixed(u3*dx_)