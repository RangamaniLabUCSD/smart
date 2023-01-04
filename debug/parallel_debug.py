import dolfin as d

import stubs
from stubs.common import _fancy_print as fancy_print

rank = d.MPI.comm_world.Get_rank()

import stubs_model

model = stubs_model.make_model(refined_mesh=False)


model._init_1()
model._init_2()
model._init_3()

print(f"processor {rank}: {model.parent_mesh.num_vertices} parent verts")
print(f"processor {rank}: {model.child_meshes['cytosol'].num_vertices} cyto verts")
print(f"processor {rank}: {model.child_meshes['er_vol'].num_vertices} ervol verts")
print(f"processor {rank}: {model.child_meshes['er_mem'].num_vertices} ermem verts")
fancy_print(
    f"{model.child_meshes['er_mem'].num_vertices} ermem verts",
    format_type="log_urgent",
)


# TODO: need to change _active_compartments (https://github.com/justinlaughlin/stubs/issues/36)
model._init_4_1_get_active_compartments()
fancy_print(f"{model.cc.sort_by('num_dofs')[1]} num_dofs", format_type="log_urgent")

model._init_4_2_define_dolfin_function_spaces()
# compartment = list(model._active_compartments)[0]
# print(f"processor {rank}: child_meshes:            {id(model.child_meshes['cytosol'].dolfin_mesh)}")
# print(f"processor {rank}: compartment.dolfin_mesh: {id(compartment.dolfin_mesh)}")
# print(f"processor {rank}: compartment:             {id(compartment)}")
# #assert model.child_meshes['cytosol'].dolfin_mesh is compartment.dolfin_mesh #fails in parallel
# V = d.FunctionSpace(model.child_meshes['cytosol'].dolfin_mesh, 'P', 1) # This works
# V2 = d.VectorFunctionSpace(model.child_meshes['cytosol'].dolfin_mesh, 'P', 1, dim=2) # This works
# W = d.MixedFunctionSpace(*[V,V2])
# #V2 = d.FunctionSpace(compartment.dolfin_mesh, 'P', 1) # These cause fenics to hang


# V = d.FunctionSpace(model.parent_mesh.dolfin_mesh, 'P', 1) # this is fine
# V = d.VectorFunctionSpace(compartment.dolfin_mesh, 'P', 1, dim=compartment.num_species)

# model._init_4_3_define_dolfin_functions()
# model._init_4_4_get_species_u_v_V_dofmaps()
# model._init_4_5_name_functions()
# model._init_4_6_check_dolfin_function_validity()
# model._init_4_7_set_initial_conditions()


print("done")


# from pathlib import Path
# path    = Path('.').resolve()
# subdir  = 'data'
# while True:
#    if path.parts[-1]=='stubs' and path.joinpath(subdir).is_dir():
#        path = path.joinpath(subdir)
#        break
#    path = path.parent
# mesh = d.Mesh(str(path / 'adjacent_cubes_refined.xml'))

# mf3 = d.MeshFunction('size_t', mesh, 3, value=mesh.domains())

# mf2 = d.MeshFunction('size_t', mesh, 2, value=mesh.domains())

# # try unmarking some of mf2(4)
# d.CompiledSubDomain('x[0] < 0.0').mark(mf2,6)

# mesh1 = d.MeshView.create(mf3, 11)
# mesh2 = d.MeshView.create(mf3, 12)
# mesh_ = d.MeshView.create(mf2, 4)

# #print(f"mesh1 mpi rank {mesh1.mpi_comm().Get_rank()}")

# V0 = d.FunctionSpace(mesh,"P",1); V1 = d.FunctionSpace(mesh1,"P",1);
# V2 = d.FunctionSpace(mesh2,"P",1); V_ = d.FunctionSpace(mesh_,"P",1);
# V  = d.MixedFunctionSpace(V0,V1,V2,V_)

# u = d.Function(V)
# u0 = u.sub(0); u1 = u.sub(1); u2 = u.sub(2); u_ = u.sub(3)
# u0.vector()[:] = 1; u1.vector()[:] = 5; u2.vector()[:] = 3; u_.vector()[:] = 7


# #print(f"processor {rank}: mesh_:                   {id(mesh_)}")

# print('finished')
