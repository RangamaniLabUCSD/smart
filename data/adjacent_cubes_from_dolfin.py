import dolfin as d

def make_adjacent_cubes_mesh(N, hdf5_filename):
    mesh = d.UnitCubeMesh(N, N, N)
    mesh.coordinates()[:,0] = mesh.coordinates()[:,0]*2 - 1
    mesh.coordinates()[:,1] = mesh.coordinates()[:,1]*2 - 1
    mesh.coordinates()[:,2] = mesh.coordinates()[:,2]*4 - 2

    mf2 = d.MeshFunction('size_t', mesh, 2, value=0)
    mf3 = d.MeshFunction('size_t', mesh, 3, value=0)
    d.CompiledSubDomain('on_boundary').mark(mf2, 2)    
    d.CompiledSubDomain('on_boundary && near(x[2], 2)').mark(mf2, 0)     # mark some facets near pm as 0 
    d.CompiledSubDomain('near(x[2], 0)').mark(mf2, 4)
    d.CompiledSubDomain('x[2] <= 0').mark(mf3, 12)
    d.CompiledSubDomain('x[2] >= 0').mark(mf3, 11)

    # write out
    hdf5 = d.HDF5File(mesh.mpi_comm(), hdf5_filename, 'w')
    hdf5.write(mesh, '/mesh')
    # write mesh functions
    hdf5.write(mf2, f"/mf2")
    hdf5.write(mf3, f"/mf3")
    hdf5.close()

# make_adjacent_cubes_mesh(20, 'adjacent_cubes_from_dolfin_10.h5')
make_adjacent_cubes_mesh(20, 'adjacent_cubes_from_dolfin_20_lessPM.h5')
# make_adjacent_cubes_mesh(20, 'adjacent_cubes_from_dolfin_30.h5')
# make_adjacent_cubes_mesh(40, 'adjacent_cubes_from_dolfin_40.h5')
# make_adjacent_cubes_mesh(60, 'adjacent_cubes_from_dolfin_60.h5')
# make_adjacent_cubes_mesh(100, 'adjacent_cubes_from_dolfin_100.h5')