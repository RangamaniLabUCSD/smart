# profiling speed with/without compile

from dolfin import *
import cProfile

timer = Timer() 
timer.start()

def func():
    #Create mesh and define function space
    mesh = UnitSquareMesh(500, 500)

    marker = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    for c in cells(mesh):
        marker[c] = c.midpoint().x() < 0.5

    submesh1 = MeshView.create(marker, 1)
    submesh2 = MeshView.create(marker, 0)

    # Define Dirichlet boundary
    def boundarySub1(x):
        return x[0] < DOLFIN_EPS

    def boundarySub2(x):
        return x[0] > 1.0 - DOLFIN_EPS

    W1 = FunctionSpace(submesh1, "Lagrange", 1)
    W2 = FunctionSpace(submesh2, "Lagrange", 1)

    # Define the mixed function space
    V = MixedFunctionSpace( W1, W2 )

    # Define boundary conditions
    u0 = Constant(0.0)
    # Subdomain 1
    bc1 = DirichletBC(V.sub_space(0), u0, boundarySub1)
    # Subdomain 2
    bc2 = DirichletBC(V.sub_space(1), u0, boundarySub2)

    # Define variational problem
    # Use directly TrialFunction and TestFunction on the product space
    (u1,u2) = TrialFunctions(V)
    (v1,v2) = TestFunctions(V)

    f = Expression("6.8*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)

    # Define new measures so that we can sort the integrals
    a_prod = inner(grad(u1), grad(v1))*dx + inner(grad(u2), grad(v2))*dx
    L_prod = f*v1*dx + f*v2*dx
    # Subdomain 1
    a1 = extract_blocks(a_prod,0,0)
    L1 = extract_blocks(L_prod,0)
    sol1 = Function(V.sub_space(0))
    solve(a1 == L1, sol1, bc1)
    # Subdomain 2
    a2 = extract_blocks(a_prod,1,1)
    L2 = extract_blocks(L_prod,1)
    sol2 = Function(V.sub_space(1))
    solve(a2 == L2, sol2, bc2)
    #print(sol2.vector().get_local().max())

func()
#cProfile.run("func()")
print(timer.stop())
