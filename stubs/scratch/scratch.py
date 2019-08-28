# scratch.py
# Code that has since been refactored or is no longer needed but may be useful as a reference...




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

