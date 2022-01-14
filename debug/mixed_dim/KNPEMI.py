from dolfin import *
import numpy as np

from KNPEMI_MMS import MMS

# Export results
do_export = False
# Polynomial order (= 1 or 2)
order = 1

def setup_square_domain(n):
    """ Inner (interior) is [0.25, 0.75]^2, outer (exterior) is
    [0, 1]^2 \ [0.25, 0.75]^2 and \partial [0.25, 0.75]^2 is the interface """

    # square mesh
    mesh = UnitSquareMesh(n, n)
    # define interior domain
    interior = CompiledSubDomain('std::max(fabs(x[0] - 0.5), fabs(x[1] - 0.5)) < 0.25')
    # create mesh function
    subdomains = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)

    # mark interior and exterior domain
    for cell in cells(mesh):
        x = cell.midpoint().array()
        subdomains[cell] = int(interior.inside(x, False))
    assert sum(1 for _ in SubsetIterator(subdomains, 1)) > 0

    # create exterior mesh
    exterior_mesh = MeshView.create(subdomains, 0)
    # create interior mesh
    interior_mesh = MeshView.create(subdomains, 1)

    # create interface mesh
    surfaces = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    for f in facets(mesh):
        interface_left = (0.25 - DOLFIN_EPS < f.midpoint().x() < 0.25 + DOLFIN_EPS) and (0.25 - DOLFIN_EPS < f.midpoint().y() < 0.75 + DOLFIN_EPS)
        interface_right = (0.75 - DOLFIN_EPS < f.midpoint().x() < 0.75 + DOLFIN_EPS) and (0.25 - DOLFIN_EPS < f.midpoint().y() < 0.75 + DOLFIN_EPS)
        interface_bottom = (0.25 - DOLFIN_EPS < f.midpoint().y() < 0.25 + DOLFIN_EPS) and (0.25 - DOLFIN_EPS < f.midpoint().x() < 0.75 + DOLFIN_EPS)
        interface_top = (0.75 - DOLFIN_EPS < f.midpoint().y() < 0.75 + DOLFIN_EPS) and (0.25 - DOLFIN_EPS < f.midpoint().x() < 0.75 + DOLFIN_EPS)
        surfaces[f] = interface_left or interface_right or interface_bottom or interface_top
    gamma_mesh = MeshView.create(surfaces, 1)

    # Mark the outside of exterior mesh
    facet_f = MeshFunction('size_t', exterior_mesh, exterior_mesh.topology().dim()-1, 0)
    CompiledSubDomain('near(x[0]*(1-x[0]), 0) || near(x[1]*(1-x[1]), 0)').mark(facet_f, 1)
    exterior_mesh.subdomains = facet_f

    return interior_mesh, exterior_mesh, gamma_mesh

def solve_system(n, t, dt, Tstop, ion_list, M, order):

    """ Solve the system """
    # get mesh
    interior_mesh, exterior_mesh, gamma_mesh = setup_square_domain(n)

    dt_inv = Constant(1./dt)    # invert time step
    dt = Constant(dt)           # make dt Constant
    # get physical parameters
    psi = 1.0
    F = 1.0
    tmp = 1.0
    R = 1.0
    C_M = 1.0

    N_ions = len(ion_list)

    # get MMS terms and exact solutions
    src_terms, exact_sols, init_conds, bndry_terms, \
                subdomains_MMS =  M.get_MMS_terms_KNPEMI(t)

    # ------------------------- Setup function spaces ----------------------- #
    # Element over interior mesh
    P1 = FiniteElement('P', interior_mesh.ufl_cell(), 1)    # ion concentrations and potentials
    P2 = FiniteElement('P', interior_mesh.ufl_cell(), 2)
    P3 = FiniteElement('P', interior_mesh.ufl_cell(), 3)
    Pk = [P1,P2,P3]
    R0 = FiniteElement('R', interior_mesh.ufl_cell(), 0)    # Lagrange to enforce /int phi_i = 0
    # Element over gamma mesh
    Q1 = FiniteElement('P', gamma_mesh.ufl_cell(), 1)        # membrane ion channels
    Q2 = FiniteElement('P', gamma_mesh.ufl_cell(), 2)
    Q3 = FiniteElement('P', gamma_mesh.ufl_cell(), 3)
    Qk = [Q1,Q2,Q3]
    
    # Intracellular ion concentrations for each ion (N_ion), potential, Lagrange multiplier
    interior_element_list = [Pk[order-1]]*(N_ions + 1) + [R0]
    # Extracellular ion concentrations for each (N_ion), potential
    exterior_element_list = [Pk[order-1]]*(N_ions + 1)

    # Create function spaces
    Wi = FunctionSpace(interior_mesh, MixedElement(interior_element_list))
    We = FunctionSpace(exterior_mesh, MixedElement(exterior_element_list))
    Wg = FunctionSpace(gamma_mesh, Qk[order-1])
    W = MixedFunctionSpace(Wi, We, Wg)
    
    # mark exterior subdomain - subdomains_MMS[0] = (x=0) U (x=1) U (y=0) U (y=1)
    exterior_subdomains = subdomains_MMS[0]
    exterior_boundary = MeshFunction('size_t', exterior_mesh, exterior_mesh.topology().dim()-1, 0)
    [subd.mark(exterior_boundary, 1) for subd in map(CompiledSubDomain, exterior_subdomains)]
    # normal on exterior boundary
    n_outer = FacetNormal(exterior_mesh)
    # measure on exterior boundary
    dsOuter = Measure('ds', domain=exterior_mesh, subdomain_data=exterior_boundary, subdomain_id=1)
    
    # mark interface - subdomains_MMS[1] = (x=0.25) U (x=0.75) U (y=0.25) U (y=0.75)
    gamma_subdomains = subdomains_MMS[1]
    # Mark interface
    gamma_boundary = MeshFunction('size_t', gamma_mesh,gamma_mesh.topology().dim(), 0)
    [subd.mark(gamma_boundary, i) for i, subd in enumerate(map(CompiledSubDomain, gamma_subdomains), 1)]  
    # measures on exterior mesh
    dxe = Measure('dx', domain=exterior_mesh)    
    # measure on interior mesh
    dxi = Measure('dx', domain=interior_mesh)
    # measure on gamma
    dxGamma = Measure('dx', domain=gamma_mesh, subdomain_data=gamma_boundary)

    # ------------------------- Setup functions ----------------------------- #
    # create functions
    # Element over interior mesh
    # P1 = FiniteElement('P', interior_mesh.ufl_cell(), 1)    # ion concentrations and potentials
    # P2 = FiniteElement('P', interior_mesh.ufl_cell(), 2)
    # P3 = FiniteElement('P', interior_mesh.ufl_cell(), 3)
    # Pk = [P1,P2,P3]
    # R0 = FiniteElement('R', interior_mesh.ufl_cell(), 0)    # Lagrange to enforce /int phi_i = 0
    # # Element over gamma mesh
    # Q1 = FiniteElement('P', gamma_mesh.ufl_cell(), 1)        # membrane ion channels
    # Q2 = FiniteElement('P', gamma_mesh.ufl_cell(), 2)
    # Q3 = FiniteElement('P', gamma_mesh.ufl_cell(), 3)
    # Qk = [Q1,Q2,Q3]
    
    # # Intracellular ion concentrations for each ion (N_ion), potential, Lagrange multiplier
    # interior_element_list = [Pk[order-1]]*(N_ions + 1) + [R0]
    # # Extracellular ion concentrations for each (N_ion), potential
    # exterior_element_list = [Pk[order-1]]*(N_ions + 1)
    # Wi = FunctionSpace(interior_mesh, MixedElement(interior_element_list))
    # We = FunctionSpace(exterior_mesh, MixedElement(exterior_element_list))
    # Wg = FunctionSpace(gamma_mesh, Qk[order-1])
    # W = MixedFunctionSpace(Wi, We, Wg)

    # e.g. Nions = 2
    # Wi = [ion_1, ion_2, V, lambda]
    # We = [ion_1, ion_2, V]
    # Wg = [mion_1]
    # W = [Wi, We, Wg]  - lengths are (nion+2), (nion+1), (1)
    (ui, ue, p_IM) = TrialFunctions(W)
    (vi, ve, q_IM) = TestFunctions(W)
    u_p = Function(W)
    ui_p = u_p.sub(0)
    ue_p = u_p.sub(1)

    # split unknowns
    ui = split(ui)
    ue = split(ue)
    # split test functions
    vi = split(vi)
    ve = split(ve)
    # split previous solution
    ui_prev = split(ui_p)
    ue_prev = split(ue_p)

    # intracellular potential
    phi_i = ui[N_ions]              # unknown
    vphi_i = vi[N_ions]             # test function
    # extracellular potential
    phi_e = ue[N_ions]              # unknown
    vphi_e = ve[N_ions]             # test function
    # Lagrange multiplier for /int phi_i = 0
    _c = ui[N_ions+1]               # unknown
    _d = vi[N_ions+1]               # test function

    # get MMS terms and exact solutions
    src_terms, exact_sols, init_conds, bndry_terms, \
                subdomains_MMS =  M.get_MMS_terms_KNPEMI(t)
    # set initial membrane potential
    phi_M_init = init_conds['phi_M']
    phi_M_prev = interpolate(phi_M_init, Wg)
    
    # --------------------- Setup variational formulation ---------------------- #
    # sum of fractions
    alpha_i_sum = 0  # intracellular
    alpha_e_sum = 0  # extracellular
    I_ch = 0         # total channel current

    # Initialize parts of variational formulation
    for idx, ion in enumerate(ion_list):
        # get ion attributes
        z = ion['z']; Di = ion['Di']; De = ion['De'];
        
        # set initial value of intra and extracellular ion concentration
        assign(ui_p.sub(idx), interpolate(ion['ki_init'], Wi.sub(idx).collapse()))
        assign(ue_p.sub(idx), interpolate(ion['ke_init'], We.sub(idx).collapse()))
        # add ion specific contribution to fraction alpha
        ui_prev_g = interpolate(ui_p.sub(idx), Wg)
        ue_prev_g = interpolate(ue_p.sub(idx), Wg)
        alpha_i_sum += Di*z*z*ui_prev_g
        alpha_e_sum += De*z*z*ue_prev_g
        
        # calculate and update Nernst potential for current ion
        ion['E'] = project(R*tmp/(F*z)*ln(ue_prev_g/ui_prev_g), Wg)
        # ion specific channel current
        ion['I_ch'] = phi_M_prev
        # add contribution to total channel current
        I_ch += ion['I_ch']

    J_phi_i = 0     # total intracellular flux
    J_phi_e = 0     # total extracellular flux

    # Initialize all parts of the variational form
    a00 = 0; a01 = 0; a02 = 0; L0 = 0
    a10 = 0; a11 = 0; a12 = 0; L1 = 0
    a20 = 0; a21 = 0; a22 = 0; L2 = 0

    # Setup ion specific part of variational formulation
    for idx, ion in enumerate(ion_list):
        # get ion attributes
        z = ion['z']; Di = ion['Di']; De = ion['De']; I_ch_k = ion['I_ch']

        # Set intracellular ion attributes
        ki = ui[idx]             # unknown
        ki_prev = ui_prev[idx]   # previous solution
        vki = vi[idx]            # test function
        # Set extracellular ion attributes
        ke = ue[idx]             # unknown
        ke_prev = ue_prev[idx]   # previous solution
        vke = ve[idx]            # test function
        # fraction of ion specific intra--and extracellular I_cap
        # Interpolate the previous solution on Gamma
        ki_prev_g = interpolate(ui_p.sub(idx), Wg)
        ke_prev_g = interpolate(ue_p.sub(idx), Wg)
        alpha_i = Di*z*z*ki_prev_g/alpha_i_sum
        alpha_e = De*z*z*ke_prev_g/alpha_e_sum

        # ion fluxes
        Ji = - Constant(Di)*grad(ki) - Constant(Di*z/psi)*ki_prev*grad(phi_i)  # linearised
        Je = - Constant(De)*grad(ke) - Constant(De*z/psi)*ke_prev*grad(phi_e)  # linearised
        
        # eq for k_i
        a00 += dt_inv*ki*vki*dxi - inner(Ji, grad(vki))*dxi
        a02 -= 1.0/(F*z)*alpha_i*p_IM*vki*dxGamma
        L0  += dt_inv*ki_prev*vki*dxi + 1.0/(F*z)*(I_ch_k - alpha_i*I_ch)*vki*dxGamma

        # eq for k_e
        a11 += dt_inv*ke*vke*dxe - inner(Je, grad(vke))*dxe
        a12 += 1.0/(F*z)*alpha_e*p_IM*vke*dxGamma
        L1  += dt_inv*ke_prev*vke*dxe - 1.0/(F*z)*(I_ch_k - alpha_e*I_ch)*vke*dxGamma

        # add contribution to total current flux
        J_phi_i += F*z*Ji
        J_phi_e += F*z*Je

        # MMS: add source terms
        L0 += inner(ion['f_k_i'], vki)*dxi # eq for k_i
        L1 += inner(ion['f_k_e'], vke)*dxe # eq for k_e
        # exterior boundary terms (zero in "physical" problem)
        L1 -= inner(dot(ion['J_k_e'], n_outer), vke)*dsOuter # eq for k_e
        L1 += F*z*inner(dot(ion['J_k_e'], n_outer), vphi_e)*dsOuter # eq for phi_e

    # equation for phi_i
    a00 += inner(J_phi_i, grad(vphi_i))*dxi
    a02 += inner(p_IM, vphi_i)*dxGamma

    # /int phi_i = 0: Lagrange terms
    a00 += _c*vphi_i*dxi + _d*phi_i*dxi

    # equation for phi_e
    a11 += inner(J_phi_e, grad(vphi_e))*dxe
    a12 -= inner(p_IM, vphi_e)*dxGamma

    # phi_M: Lagrange terms
    a20 += inner(phi_i, q_IM)*dxGamma
    a21 -= inner(phi_e, q_IM)*dxGamma
    a22 -= dt/C_M*inner(p_IM, q_IM)*dxGamma
    L2  += inner(phi_M_prev, q_IM)*dxGamma \
         - dt/C_M*inner(I_ch, q_IM)*dxGamma

    # add source term if MMS test
    L0 += inner(ion['phi_i_e'], _d)*dxi  # Lagrange for phi_i (if int phi_I != 0)
    L0 += inner(ion['f_phi_i'], vphi_i)*dxi    # eq for phi_i
    L1 += inner(ion['f_phi_e'], vphi_e)*dxe    # eq for phi_e

    # coupling condition IM = -Ji = Je + g
    L1 -= sum(inner(gM, vphi_e)*dxGamma(i) for i, gM in enumerate(ion['f_g_M'], 1))
    # eq for J_M
    L2 += dt/C_M*sum(inner(JM, q_IM)*dxGamma(i) for i, JM in enumerate(ion['f_J_M'], 1))
    
    # gather var form in matrix structure
    a = a00 + a01 + a02 + a10 + a11 + a12 + a20 + a21 + a22
    L = L0 + L1 + L2

    # -------------------------------- Solve ----------------------------------- #
    for k in range(int(round(Tstop/float(dt)))):

        wh = Function(W)
        solve(a == L, wh, solver_parameters={"linear_solver":"direct"})

        ui_p.assign(wh.sub(0))  # update ion specific membrane channels
        ue_p.assign(wh.sub(1))  # update ion specific membrane channels
        u_p.sub(2).assign(wh.sub(2))
        
        ## Export
        if do_export:
            encoding = XDMFFile.Encoding.HDF5 if has_hdf5() else XDMFFile.Encoding.ASCII
            if MPI.size(MPI.comm_world) > 1 and encoding == XDMFFile.Encoding.ASCII:
                print("XDMF file output not supported in parallel without HDF5")
            for i in range(N_ions+1):
                out_subi = XDMFFile(MPI.comm_world, "test-i"+str(i)+".xdmf")
                out_subi.write(ui_p.sub(i), encoding)
                out_sube = XDMFFile(MPI.comm_world, "test-e"+str(i)+".xdmf")
                out_sube.write(ue_p.sub(i), encoding)
            out_subg = XDMFFile(MPI.comm_world, "test-g.xdmf")
            out_subg.write(u_p.sub(2))

        # update previous membrane potential
        phi_M_prev.assign(interpolate(ui_p.sub(N_ions), Wg) \
                          - interpolate(ue_p.sub(N_ions), Wg))

        # updates problems time t
        t.assign(float(t + dt))

        # update Nernst potential for all ions
        for idx, ion in enumerate(ion_list):
            z = ion['z']
            ke_prev_g = interpolate(ue_p.sub(idx), Wg)
            ki_prev_g = interpolate(ui_p.sub(idx), Wg)
            ion['E'] = R*tmp/(F*z)*ln(ke_prev_g/ki_prev_g)

        """
        # DEBUG
        print("E_K", ion_list[1]['E'])
        print("E_Na", ion_list[0]['E'])
        print("total interior Na ", assemble(wh[0].sub(0)*dx))
        print("total exterior Na ", assemble(wh[1].sub(0)*dx))
        print("total Na ", assemble(wh[1].sub(0)*dx) + assemble(wh[0].sub(0)*dx))
        intrap = (0.35, 0.35)
        diff = project(wh[0].sub(0)(intrap) + wh[0].sub(1)(intrap) - wh[0].sub(2)(intrap), trace_space)
        print(diff.vector().array())
        """

    return wh, interior_mesh, exterior_mesh, gamma_mesh
