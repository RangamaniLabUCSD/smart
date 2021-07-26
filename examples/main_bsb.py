# Example showing bulk-surface-bulk problem

# import 
import dolfin as d # dolfin/fenics api
#import mpi4py.MPI as pyMPI
import stubs
import os
from stubs import unit as ureg

# ====================================================
# ====================================================
comm = d.MPI.comm_world
rank = comm.rank
root = 0
nprocs = comm.size
cwd = os.getcwd()

# Load in model and settings
config = stubs.config.Config()
config.reaction_database.update({'leak_dynamic' : 'perm*(uhigh-ulow)'})

PD, SD, CD, RD = stubs.common.read_smodel('bsb/cell2013_3d.smodel')

# Define solvers
mps = stubs.solvers.MultiphysicsSolver('iterative', eps_Fabs=1e-8)
nls = stubs.solvers.NonlinearNewtonSolver(relative_tolerance=1e-6, absolute_tolerance=1e-8,
                                          dt_increase_factor=1.05, dt_decrease_factor=0.7)
ls = stubs.solvers.DolfinKrylovSolver(method = 'bicgstab', preconditioner='hypre_amg')
solver_system = stubs.solvers.SolverSystem(final_t = 0.4, initial_dt = 0.01, adjust_dt = [(0.2, 0.02)],
                                           multiphysics_solver=mps, nonlinear_solver=nls, linear_solver=ls)

main_mesh = stubs.mesh.Mesh(mesh_filename=cwd+'/bsb/cube_10.xml')

model = stubs.model.Model(PD, SD, CD, RD, config, solver_system, main_mesh)


##======================================
#model.initialize_refactor()
##======================================
print("\n\n********** Model initialization (Part 1/6) **********")
print("Assembling parameters and units...\n")
model.PD.do_to_all('assemble_units', {'unit_name': 'unit'})
model.PD.do_to_all('assemble_units', {'value_name':'value', 'unit_name':'unit', 'assembled_name': 'value_unit'})
model.PD.do_to_all('assembleTimeDependentParameters')
model.SD.do_to_all('assemble_units', {'unit_name': 'concentration_units'})
model.SD.do_to_all('assemble_units', {'unit_name': 'D_units'})
model.CD.do_to_all('assemble_units', {'unit_name':'compartment_units'})
model.RD.do_to_all('initialize_flux_equations_for_known_reactions', {"reaction_database": model.config.reaction_database})

# linking containers with one another
print("\n\n********** Model initialization (Part 2/6) **********")
print("Linking different containers with one another...\n")
model.RD.link_object(model.PD,'paramDict','name','paramDictValues', value_is_key=True)
model.SD.link_object(model.CD,'compartment_name','name','compartment')
model.SD.copy_linked_property('compartment', 'dimensionality', 'dimensionality')
model.RD.do_to_all('get_involved_species_and_compartments', {"SD": model.SD, "CD": model.CD})
model.RD.link_object(model.SD,'involved_species','name','involved_species_link')

# meshes
print("\n\n********** Model initialization (Part 3/6) **********")
print("Loading in mesh and computing statistics...\n")
model.CD.get_min_max_dim()
setattr(model.CD, 'meshes', {model.mesh.name: model.mesh.mesh})
#model.CD.extract_submeshes('cyto', save_to_file=False)
model.CD.extract_submeshes_refactor(save_to_file=False)
# model.CD.compute_scaling_factors()

# # Associate species and compartments
# print("\n\n********** Model initialization (Part 4/6) **********")
# print("Associating species with compartments...\n")
# num_species_per_compartment = model.RD.get_species_compartment_counts(model.SD, model.CD)
# model.SD.assemble_compartment_indices(model.RD, model.CD)
# model.CD.add_property_to_all('is_in_a_reaction', False)
# model.CD.add_property_to_all('V', None)

# # dolfin functions
# print("\n\n********** Model initialization (Part 5/6) **********")
# print("Creating dolfin functions and assinging initial conditions...\n")
# model.SD.assemble_dolfin_functions(model.RD, model.CD)
# model.u = model.SD.u
# model.v = model.SD.v
# model.V = model.SD.V
# model.assign_initial_conditions()

# print("\n\n********** Model initialization (Part 6/6) **********")
# print("Assembling reactive and diffusive fluxes...\n")
# model.RD.reaction_to_fluxes()
# #model.RD.do_to_all('reaction_to_fluxes')
# model.FD = model.RD.get_flux_container()
# model.FD.do_to_all('get_additional_flux_properties', {"CD": model.CD, "solver_system": model.solver_system})
# model.FD.do_to_all('flux_to_dolfin')

# model.set_allow_extrapolation()
# # Turn fluxes into fenics/dolfin expressions
# model.assemble_reactive_fluxes()
# model.assemble_diffusive_fluxes()
# model.sort_forms()

# model.init_solutions_and_plots()
##======================================


# solve system
#model.solve(store_solutions=False)
#model.solve_zero_d([0,1])
