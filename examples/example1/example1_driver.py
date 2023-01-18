# =============================================================================================
# STUBS driver script example
# =============================================================================================
# Imports
import os

import dolfin as d

import example1_model

import stubs


# =============================================================================================
# Load model
# =============================================================================================
pc, sc, cc, rc = example1_model.make_model()

# =============================================================================================
# Create/load in mesh
# =============================================================================================
# Base mesh
mesh, mf2, mf3 = stubs.common.DemoCuboidsMesh()
# Turn off "PM" on all sides of the cube except x=0
for face in d.faces(mesh):
    if face.midpoint().x() > d.DOLFIN_EPS and mf2[face]==10:
        mf2[face] = 0
# Write mesh and meshfunctions to file
os.makedirs('mesh', exist_ok=True)
stubs.common.write_mesh(mesh, mf2, mf3, filename='mesh/DemoCuboidsMesh')

# # Define solvers
parent_mesh = stubs.mesh.ParentMesh(mesh_filename='mesh/DemoCuboidsMesh.h5', mesh_filetype='hdf5', name='parent_mesh')
config = stubs.config.Config()
model = stubs.model.Model(pc, sc, cc, rc, config, parent_mesh)
config.solver.update({'final_t':1, 'initial_dt':.01, 'time_precision': 6, 'use_snes': True, 'print_assembly': False})
model.initialize(initialize_solver=False)
model.initialize_discrete_variational_problem_and_solver()

# Write initial condition(s) to file
results = dict()
os.makedirs('results', exist_ok=True)
for species_name, species in model.sc.items:
    results[species_name] = d.XDMFFile(model.mpi_comm_world, f'results/{species_name}.xdmf')
    results[species_name].parameters['flush_output'] = True
    results[species_name].write(model.sc[species_name].u['u'], model.t)

# Solve
while True:
    # Solve the system
    model.monolithic_solve()
    # Save results for post processing
    for species_name, species in model.sc.items:
        results[species_name].write(model.sc[species_name].u['u'], model.t)
    # End if we've passed the final time 
    if model.t >= model.final_t:
        break
