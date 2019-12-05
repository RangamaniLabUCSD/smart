# scratch.py
"""
Bits of code which should be culled but may have something worth extracting
"""


# from model_assembly.py


# # ====================================================
# # Parameters

# def assemble_parameters(pdf, value, unitStr, newColumnName):
#     ## Add a column to a dataframe which contains pint "quantity" objects
#     pdf[newColumnName] = pd.Series(dtype=object)
#     for idx, param in pdf.iterrows():
#         pdf.at[idx,newColumnName] = param.value * unit(param[unitStr])
#         if param.is_time_dependent:
#             pdf.at[idx,'symExpr'] = parse_expr(param.symExpr)
#             print("Creating dolfin object for time-dependent parameter %s" % param.parameter_name)
#             pdf.at[idx,'dolfinConstant'] = ref(d.Constant(param.value))

# # ====================================================
# # Species

# def assemble_units(df, unitList):
#     if not isinstance(unitList, list): unitList = [unitList]
#     for unitStr in unitList:
#         for idx, row in df.iterrows():
#             df.at[idx, unitStr] = unit(row[unitStr])

# def assemble_dolfin_functions(rdf, sdf, cdf, meshes):
#     ## define dof/solution vectors (dolfin trialfunction, testfunction, and function types) based on number of species appearing in reactions
#     # obtain a list of unique species
#     num_species_per_compartment = get_species_compartment_counts(rdf, sdf)
#     # create the function space, functions, and test functions
#     V, u, v = {}, {}, {}
#     for compartment, num_species in num_species_per_compartment.items():
#         compartmentDim = cdf.loc[cdf.compartment_name==compartment,'dimensionality'].squeeze()
#         cdf.loc[cdf.compartment_name==compartment,'num_species'] = num_species
#         print('Compartment %s (dimension: %d) has %d species associated with it' % (compartment, compartmentDim, num_species))

#         # u is the actual function. t is for linearized versions. k is for picard iterations. n is for last time-step solution
#         if num_species == 1:
#             V[compartment] = d.FunctionSpace(meshes[compartment], 'P', 1)
#             u[compartment] = {'u': d.Function(V[compartment]), 't': d.TrialFunction(V[compartment]),
#             'k': d.Function(V[compartment]), 'n': d.Function(V[compartment])}
#             v[compartment] = d.TestFunction(V[compartment])
#         else: # vector space
#             V[compartment] = d.VectorFunctionSpace(meshes[compartment], 'P', 1, dim=num_species)
#             u[compartment] = {'u': d.Function(V[compartment]), 't': d.TrialFunctions(V[compartment]),
#             'k': d.Function(V[compartment]), 'n': d.Function(V[compartment])}
#             v[compartment] = d.TestFunctions(V[compartment])

#     # now we create boundary functions, which are defined on the function spaces of the surrounding mesh
#     V['boundary'] = {}
#     #V.update({'boundary': {}})
#     for compartment, num_species in num_species_per_compartment.items():
#         compartmentDim = cdf.loc[cdf.compartment_name==compartment,'dimensionality'].squeeze()
#         if compartmentDim == 3: # mesh may have boundaries
#             for boundary_name, boundary_mesh in meshes.items():
#                 if compartment != boundary_name:
#                     if num_species == 1:
#                         boundaryV = d.FunctionSpace(meshes[boundary_name], 'P', 1)
#                     else:
#                         boundaryV = d.VectorFunctionSpace(meshes[boundary_name], 'P', 1, dim=num_species)
#                     V['boundary'].update({compartment: {boundary_name: boundaryV}})
#                     u[compartment].update({'b': d.Function(boundaryV)})


#     assemble_compartment_indices(rdf, sdf)

#     # associate indexed functions with dataframe
#     #keys = ['u', 't', 'k', 'n']
#     sdf['u'] = pd.Series(dtype=object)
#     sdf['v'] = pd.Series(dtype=object)
#     cdf['is_in_a_reaction'] = False
#     for idx, row in sdf.iterrows():
#         sdf.at[idx,'u'] = {}
#         if row.is_in_a_reaction:
#             cdf.loc[cdf.compartment_name==row.compartment_name, 'is_in_a_reaction'] = True
#             num_species = cdf.loc[cdf.compartment_name==row.compartment_name,'num_species'].squeeze()
#             for key in u[row.compartment_name].keys():
#                 if num_species == 1:
#                     sdf.at[idx,'u'].update({key: ref(u[row.compartment_name][key])})
#                     sdf.at[idx,'v'] = ref(v[row.compartment_name])
#                 else:
#                     sdf.at[idx,'u'].update({key: ref(u[row.compartment_name][key][row.compartment_index])})
#                     sdf.at[idx,'v'] = ref(v[row.compartment_name][row.compartment_index])


#     # associate function spaces with dataframe
#     cdf['V'] = pd.Series(dtype=object)
#     for idx, row in cdf.iterrows():
#         if row.is_in_a_reaction:
#             cdf.at[idx,'V'] = ref(V[row.compartment_name])


#     return (V, u, v)

# def assign_initial_conditions(sdf, u, V):
#     """
#     For now, these are spatially homogeneous ICs
#     """
#     keys = ['k', 'n']
#     for idx, sp in sdf.iterrows():
#         comp = sp.compartment_name

#         for key in keys:
#             data_manipulation.dolfinSetFunctionValues(u[comp][key], sp.initial_condition,
#                                        V[comp], sp.compartment_index)

#         u[comp]['u'].assign(u[comp]['n'])
#         print("Assigned initial condition to u for species %s" % sp.species_name)

#     # add boundary values
#     for comp in u.keys():
#         if 'b' in u[comp].keys():
#             u[comp]['b'].interpolate(u[comp]['u'])


# def assemble_compartment_indices(rdf, sdf):
    
#     Adds a column to the species dataframe which indicates the index of a species relative to its compartment
    
#     num_species_per_compartment = get_species_compartment_counts(rdf, sdf)
#     sdf['compartment_index'] = None # initialize column

#     for compartment, num_species in num_species_per_compartment.items():
#         index = 0
#         sdf_trunc = sdf[sdf.compartment_name==compartment].reset_index()
#         for idx, row in sdf_trunc.iterrows():
#             if row.is_in_a_reaction:
#                 sdf.loc[sdf.species_name==row.species_name, 'compartment_index'] = idx
#             else:
#                 print('Warning: species %s is not used in any reactions!' % row.species_name)


# # ====================================================
# # Compartments



# def get_species_compartment_counts(rdf, sdf):
#     """
#     Find, label, and count the number of species in each compartment involved in some reaction
#     Returns a dict like object which gives the number of species in each compartment
#     """
#     speciesList = []
#     for idx, rxn in rdf.iterrows():
#         speciesList += rxn.reactants + rxn.products

#     sdf['is_in_a_reaction'] = False # initialize column

#     speciesList = list(set(speciesList))
#     # find the compartments for each species
#     compList = []
#     for sp in speciesList:
#         comp = sdf[sdf.species_name.str.match(sp)]
#         if comp.empty:
#             raise Exception('Species %s does not have an associated compartment' % sp)
#         else:
#             compList += [comp.compartment_name.iloc[0]]
#             sdf.loc[sdf.species_name.str.match(sp), 'is_in_a_reaction'] = True

#     num_species_per_compartment = Counter(compList)

#     return num_species_per_compartment

# def merge_sdf_cdf(sdf, cdf):
#     uniqueColumns = cdf.columns.difference(sdf.columns)
#     return sdf.merge(cdf[uniqueColumns.union(['compartment_name'])], on='compartment_name')

# def extract_submeshes(meshes, cdf):
#     """
#     Extract submeshes on the boundary of a volume mesh based on boundary markers
#     """
#     # right now, hardcoded for one volumetric domain
#     surfaceDim = 2 # dimension of a surface

#     vmesh                = meshes['cyto']
#     bmesh               = d.BoundaryMesh(vmesh, "exterior") # boundary mesh
#     vmf                  = d.MeshFunction("size_t", meshes['cyto'], surfaceDim, vmesh.domains()) # mesh function on volume mesh
#     bmf                 = d.MeshFunction("size_t", bmesh, surfaceDim) # mesh function on boundary mesh
#     for idx, facet in enumerate(d.entities(bmesh,surfaceDim)): # iterate through faces of bmesh
#         vmesh_idx = bmesh.entity_map(surfaceDim)[idx] # get the index of the face on vmesh corresponding to this face on bmesh
#         vmesh_boundarynumber = vmf.array()[vmesh_idx] # get the value of the mesh function at this face
#         bmf.array()[idx] = vmesh_boundarynumber # set the value of the boundary mesh function to be the same value

#     # for now the base is always meshes['cyto']
#     cdf['mesh'] = None # initialize as object type
#     for idx, comp in cdf.iterrows():
#         if comp.compartment_name != 'cyto' and comp.dimensionality == surfaceDim:
#             meshes[comp.compartment_name] = d.SubMesh(bmesh, bmf, comp.cell_marker)

#         cdf.at[idx, 'mesh'] = meshes[comp.compartment_name]

#     return (meshes, vmf, bmf)


# def compute_volumes(cdf):
#     """
#     Computes "n-volume" (volume in 3d, area if in 2d) for each mesh
#     """
#     cdf['nvolume'] = None # initialize as object type

#     for idx, comp in cdf.iterrows():
#         cdf.loc[cdf.index[idx],'nvolume'] = d.assemble(d.Constant(1.0)*comp.dx) * comp.compartment_units ** comp.dimensionality


# def compute_scaling_factors(cdf):
#     """
#     Computes scaling factors (e.g. volume to surface area ratio) for each mesh
#     """
#     insert_dataframe_col(cdf, 'scale_to')
#     for idx, comp in cdf.iterrows():
#         for jidx, jcomp in cdf.iterrows():
#             if idx != jidx:
#                 cdf.at[idx, 'scale_to'].update({cdf.at[jidx, 'compartment_name']: cdf.at[idx,'nvolume'] / cdf.at[jidx,'nvolume']})



# def define_integration_measures(cdf, vmf):
#     for idx, comp in cdf.iterrows():
#         if comp.dimensionality == 3: # We also define a surface measure
#             measure = d.Measure('dx', domain=comp.mesh)
#             cdf.loc[cdf.index[idx], 'dx'] = measure

#             measure = d.Measure('ds', domain=comp.mesh, subdomain_data=vmf)
#             cdf.loc[cdf.index[idx], 'ds'] = measure
#         elif comp.dimensionality == 2:
#             measure = d.Measure('dP', domain=comp.mesh) # for spatially distributed ODEs
#             cdf.loc[cdf.index[idx], 'dP'] = measure

#             measure = d.Measure('dx', domain=comp.mesh)
#             cdf.loc[cdf.index[idx], 'dx'] = measure


# # ====================================================
# # Reactions

# def associate_reaction_parameters(rdf, pdf):
#     """
#     Associate the strings in rdf with the appropriate parameters in pdf
#     """
#     # initialize by inserting a new column of empty dicts
#     insert_dataframe_col(rdf, 'param', columnNumber=rdf.columns.get_loc('paramDict')+1)
#     for idx, rxn in rdf.iterrows():
#         for kinetic_parameter, value in rxn.paramDict.items():
#             if type(value) == str:
#                 rxn.param[kinetic_parameter] = pdf.loc[pdf.parameter_name==value].squeeze().value_unit
#             else:
#                 raise Exception('Parameter \"%s\" from the reaction \"%s\" does not have an associated value' % (kinetic_parameter, rxn.reaction_name))


# ====================================================
# Form assembly


# def assemble_forms_reactions(jdf, model_parameters):
#     #insert_dataframe_col(sdf, 'Form', items=0)
#     #for jidx, flux in jdf.iterrows():
#     #    sdf.loc[sdf.species_name==flux.species_name, 'Form'] += flux.flux.obj
#     Forms = {}
#     for jidx, j in jdf.iterrows():
#         comp = j.compartment_name
#         if comp not in Forms.keys():
#             Forms[comp] = {}

#         if j.flux_dimensionality == [2,3]:
#             # TODO: implement ability to apply BC to unions of compartments
#             int_measure = j.ds(j.boundary_marker)
#         elif j.flux_dimensionality == [3,3]:
#             int_measure = j.dx
#         elif j.flux_dimensionality in ([2,2], [3,2]):
#             if model_parameters['ignore_surface_diffusion']:
#                 int_measure = j.dP
#             else:
#                 int_measure = j.dx
#         jdf.loc[jidx, 'measure'] = int_measure

#         flux = j.flux.get()*j.v.get()*int_measure


#         if j.species_name in Forms[comp].keys():
#             Forms[comp][j.species_name] += -flux # minus sign because reactions are on RHS of PDE
#         else:
#             Forms[comp][j.species_name] = -flux

#     return Forms




# def assemble_forms_diffusion(sdf, dT, Forms, model_parameters):

#     for compName, comp in Forms.items():
#         for spName, sp in comp.items():
#             spdf = sdf.loc[sdf.species_name==spName].squeeze() # row of dataframe for species
#             if spdf.is_in_a_reaction:
#                 u = spdf.u['t'].get()
#                 un = spdf.u['n'].get()
#                 v = spdf.v.get()
#                 D = spdf.D
#                 dim = spdf.dimensionality

#                 if dim == 3 or model_parameters['ignore_surface_diffusion']==False:
#                     dx = spdf.dx
#                     Forms[compName][spName] += D*d.inner(d.grad(u),d.grad(v)) * dx
#                 elif dim==2 and model_parameters['ignore_surface_diffusion']:
#                     dx = spdf.dP

#                 Forms[compName][spName] += (u-un)/dT * v * dx # time derivative
#                 #(spdf.u['u'].get() - spdf.u['n'].get())/dT * spdf.v.get() * spdf.dx # time derivative
#                 #inner(grad(spdf.u['u'].get()),grad()) * spdf.dx # diffusion term after integration by parts
#             else:
#                 print("Species %s is not in a reaction?" % spName)

#     return Forms






# from Model class
        
#    def boundary_reactions_forward_scipy(self, comp_name, factor=1, all_dofs=False, method='RK45', rtol=1e-4, atol=1e-6):
#        self.stopwatch("Boundary reactions forward %s" % comp_name)
#        """
#        TODO: Either fix this or figure out how to integrate full submesh
#        parallelization
#        Since FEniCS doesn't support submeshes in parallel we wish to
#        parallelize manually. However for now this distributes the entire
#        submesh to each processor
#        """
#
#        # time step forward (irrelevant except for keeping track of time since solve_ivp() uses lambdas for time dependent parameters)
#        self.forward_time_step(factor=factor) # increment time 
#        #self.updateTimeDependentParameters() 
#
#        if all_dofs:
#            num_vertices = self.CD.Dict[comp_name].num_vertices
#        else:
#            x,y,z = (0,0,0) # point to evaluate
#            num_vertices = 1
#        if comp_name not in self.scipy_odes.keys():
#            self.scipy_odes[comp_name] = self.flux_to_scipy(comp_name, mult=num_vertices)
#        lode, ptuple, tparam, boundary_species = self.scipy_odes[comp_name]
#
#        nbspecies = len(boundary_species)
#        ub = np.full(nbspecies * num_vertices, np.nan)
#        for spidx, sp in enumerate(boundary_species):
#            pcomp_name = self.SD.Dict[sp].compartment_name
#            pcomp_idx = self.SD.Dict[sp].compartment_index
#            pcomp_nspecies = self.V['boundary'][pcomp_name][comp_name].num_sub_spaces()
#            if pcomp_nspecies==0: pcomp_nspecies=1
#            if all_dofs:
#                ub[spidx::nbspecies] = self.u[pcomp_name]['b'+comp_name].vector()[pcomp_idx::pcomp_nspecies]
#            else:
#                ub[spidx] = self.u[pcomp_name]['b'+comp_name](x,y,z)[pcomp_idx]
#                
#
#        if all_dofs:
#            sol = solve_ivp(lambda t,y: lode(t,y,ptuple,tparam,ub=ub), [self.t-self.dt*factor, self.t], self.u[comp_name]['n'].vector(), method=method, rtol=rtol, atol=atol)
#            # assign solution
#            self.u[comp_name]['u'].vector()[:] = sol.y[:,-1]
#        else:
#            # all vertices have the same value
#            sol = solve_ivp(lambda t,y: lode(t,y,ptuple,tparam,ub=ub), [self.t-self.dt*factor, self.t], self.u[comp_name]['n'](x,y,z), method=method, rtol=rtol, atol=atol)
#            for idx, val in enumerate(sol.y[:,-1]):
#                stubs.data_manipulation.dolfinSetFunctionValues(self.u[comp_name]['u'], val, idx) 
#
#
#        self.u[comp_name]['n'].assign(self.u[comp_name]['u'])
#
#        self.stopwatch("Boundary reactions forward %s" % comp_name, stop=True)
#
#
#    # TODO
#    def flux_to_scipy(self, comp_name, mult=1):
#        """
#        mult allows us to artificially make an ODE repeat e.g. 
#        dy = [dy_1, dy_2, dy_3] -> (mult=2) dy=[dy_1, dy_2, dy_3, dy_1, dy_2, dy_3]
#        Useful when we want to solve a distributed ODE on some domain so that
#        scipy can work its vector optimization magic
#        """
#        dudt = []
#        param_list = []
#        time_param_list = []
#        species_list = list(self.SD.Dict.values())
#        species_list = [s for s in species_list if s.compartment_name==comp_name]
#        species_list.sort(key = lambda s: s.compartment_index)
#        spname_list = [s.name for s in species_list]
#        num_species = len(species_list)
#
#        flux_list = list(self.FD.Dict.values())
#        flux_list = [f for f in flux_list if f.species_name in spname_list]
#
#        for idx in range(num_species):
#            sp_fluxes = [f.total_scaling*f.signed_stoich*f.symEqn for f in flux_list if f.species_name == spname_list[idx]]
#            total_flux = sum(sp_fluxes)
#            dudt.append(total_flux)
#
#            if total_flux:
#                for psym in total_flux.free_symbols:
#                    pname = str(psym)
#                    if pname in self.PD.Dict.keys():
#                        p = self.PD.Dict[pname]
#                        if p.is_time_dependent:
#                            time_param_list.append(p)
#                        else:
#                            param_list.append(pname)
#
#        
#        param_list = list(set(param_list))
#        time_param_list = list(set(time_param_list))
#
#        ptuple = tuple([self.PD.Dict[str(x)].value for x in param_list])
#        time_param_lambda = [lambdify('t', p.symExpr, modules=['sympy','numpy']) for p in time_param_list]
#        time_param_name_list = [p.name for p in time_param_list]
#
#        free_symbols = list(set([str(x) for total_flux in dudt for x in total_flux.free_symbols]))
#
#        boundary_species = [str(sp) for sp in free_symbols if str(sp) not in spname_list+param_list+time_param_name_list]
#        num_boundary_species = len(boundary_species)
#        if boundary_species:
#            Print("Adding species %s to flux_to_scipy" % boundary_species)
#        #Params = namedtuple('Params', param_list)
#
#        dudt_lambda = [lambdify(flatten(spname_list+param_list+time_param_name_list+boundary_species), total_flux, modules=['sympy','numpy']) for total_flux in dudt]
#
#
#        def lambdified_odes(t, u, p, time_p, ub=[]):
#            if int(mult*num_species) != len(u):
#                raise Exception("mult*num_species [%d x %d = %d] does not match the length of the input vector [%d]!" %
#                                (mult, num_species, mult*num_species, len(u)))
#            time_p_eval = [f(t) for f in time_p]
#            dudt_list = []
#            for idx in range(mult):
#                idx0 = idx*num_species
#                idx0b = idx*num_boundary_species
#                inp = flatten([u[idx0 : idx0+num_species], p, time_p_eval, ub[idx0b : idx0b+num_boundary_species]])
#                dudt_list.extend([f(*inp) for f in dudt_lambda])
#            return dudt_list
#
#        return (lambdified_odes, ptuple, time_param_lambda, boundary_species)
#
#
#    def IMEX_1BDF(self, method='RK45'):
#        self.stopwatch("Total time step")
#        self.idx += 1
#        Print('\n\n *** Beginning time-step %d [time=%f, dt=%f] ***\n\n' % (self.idx, self.t, self.dt))
#
#        self.boundary_reactions_forward_scipy('pm', factor=0.5, method=method, rtol=1e-5, atol=1e-8)
#        self.set_time(self.t-self.dt/2) # reset time back to t
#        self.boundary_reactions_forward_scipy('er', factor=0.5, all_dofs=True, method='RK45')
#        self.update_solution_boundary_to_volume()
#       
#
#        self.set_time(self.t-self.dt/2) # reset time back to t
#        self.IMEX_order1_diffusion_forward('cyto', factor=1)
#        self.update_solution_volume_to_boundary()
#
#        self.set_time(self.t-self.dt/2) # reset time back to t+dt/2
#        self.boundary_reactions_forward_scipy('pm', factor=0.5, method=method, rtol=1e-5, atol=1e-8)
#        self.set_time(self.t-self.dt/2) # reset time back to t+dt/2
#        self.boundary_reactions_forward_scipy('er', factor=0.5, all_dofs=True, method='RK45')
#        self.update_solution_boundary_to_volume()
#
#        if self.linear_iterations >= self.config.solver['linear_maxiter']:
#            self.set_time(self.t, dt=self.dt*self.config.solver['dt_decrease_factor'])
#            Print("Decreasing step size")
#        if self.linear_iterations < self.config.solver['linear_miniter']:
#            self.set_time(self.t, dt=self.dt*self.config.solver['dt_increase_factor'])
#            Print("Increasing step size")
#
#        self.stopwatch("Total time step", stop=True)
#
#
#    def IMEX_order1_diffusion_forward(self, comp_name, factor=1):
#        self.stopwatch("Diffusion step")
#        self.forward_time_step(factor=factor)
#        self.updateTimeDependentParameters()
#        d.parameters['form_compiler']['optimize'] = True
#        d.parameters['form_compiler']['cpp_optimize'] = True
#
#        forms = self.split_forms[comp_name]
#
#        self.stopwatch('A assembly')
#        if self.idx <= 1:
#            # terms which will not change across time-steps
#            self.Abase = d.assemble(forms['Mu'] + forms['D'], form_compiler_parameters={'quadrature_degree': 4}) # +d.lhs(forms["R"])
#            self.solver = d.KrylovSolver('cg','hypre_amg')
#            self.solver.parameters['nonzero_initial_guess'] = True
#
#
##        # if the time step size changed we need to reassemble the LHS matrix...
##        if self.idx > 1 and (self.linear_iterations >= self.config.solver['linear_maxiter'] or
##           self.linear_iterations < self.config.solver['linear_miniter']):
##            self.stopwatch('A assembly')
##            self.A = d.assemble(forms['Mu'] + forms['D'] + d.lhs(forms['R'] + d.lhs(forms['B'])), form_compiler_parameters={'quadrature_degree': 4})
##            self.stopwatch('A assembly', stop=True)
##            self.linear_iterations = 0
##            Print("Reassembling A because of change in time-step")
##
##        # sanity check to make sure A is not changing
##        if self.idx == 2:
##            Anew = d.assemble(forms['Mu'] + forms['D'] + d.lhs(forms['R'] + d.lhs(forms['B'])), form_compiler_parameters={'quadrature_degree': 4})
##            Print("Ainit linf norm = %f" % self.A.norm('linf'))
##            Print("Anew linf norm = %f" % Anew.norm('linf'))
##            assert np.abs(self.A.norm('linf') - Anew.norm('linf')) < 1e-10
#
#        # full assembly in 1 step requires using previous time step value of volumetric species for boundary fluxes
#        self.A = self.Abase + d.assemble(d.lhs(forms['B'] + forms['R']), form_compiler_parameters={'quadrature_degree': 4})
#        self.stopwatch('A assembly', stop=True)
#
#        self.stopwatch('b assembly')
#        b = d.assemble(-forms['Mun'] +  d.rhs(forms['B'] + forms['R']), form_compiler_parameters={'quadrature_degree': 4})
#        self.stopwatch('b assembly', stop=True)
#
#        U = self.u[comp_name]['u'].vector()
#        self.linear_iterations = self.solver.solve(self.A, U, b)
#
#        self.u[comp_name]['n'].assign(self.u[comp_name]['u'])
#
#        self.stopwatch("Diffusion step", stop=True)
#        Print("Diffusion step finished in %d iterations" % self.linear_iterations)
#        
#
