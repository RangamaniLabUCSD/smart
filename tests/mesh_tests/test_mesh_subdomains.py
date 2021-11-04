import stubs
import dolfin as d
import pytest

@pytest.fixture
def mesh_filename(datadir):
    return str(datadir.joinpath('adjacent_cubes.xml'))

@pytest.fixture
def dolfin_mesh(mesh_filename):
    return d.Mesh(mesh_filename)

@pytest.fixture
def stubs_mesh(mesh_filename):
    return stubs.mesh.ParentMesh(mesh_filename=mesh_filename)

@pytest.mark.stubs
def test_stubs_mesh_load_dolfin_mesh(stubs_mesh):
    "Make sure that stubs is loading the dolfin mesh when we create a ParentMesh"
    assert stubs_mesh.dolfin_mesh.num_vertices() > 1
    assert stubs_mesh.dolfin_mesh.num_cells() > 1

@pytest.mark.dolfin
def test_dolfin_cell_markers(dolfin_mesh):
    "Test if marker values (facets and cells) are loaded in from dolfin file"
    mf_3 = d.MeshFunction('size_t', dolfin_mesh, 3, value=dolfin_mesh.domains())
    # all cell markers should be either 11 or 12
    assert all([x in [11,12] for x in mf_3.array()])
    assert mf_3.array().size == dolfin_mesh.num_cells()

@pytest.mark.dolfin
def test_meshview(dolfin_mesh):
    "Test that dolfin.MeshView() is extracting submeshes correctly"
    mf_3 = d.MeshFunction('size_t', dolfin_mesh, 3, value=dolfin_mesh.domains())
    submesh_11 = d.MeshView.create(mf_3, 11)
    submesh_12 = d.MeshView.create(mf_3, 12)
    assert submesh_11.num_cells() == len(mf_3.where_equal(11))
    assert submesh_12.num_cells() == len(mf_3.where_equal(12))
    assert submesh_11.num_cells() + submesh_12.num_cells() == dolfin_mesh.num_cells()

@pytest.mark.dolfin
def test_multidim_function_space(dolfin_mesh):
    """
    Test that dolfin's multi-dimensional function spaces are behaving as expected
    This tests an uncoupled problem on two subvolumes and ensures that it is the same solution as the problem on the combined volume
    """

    mf_3        = d.MeshFunction('size_t', dolfin_mesh, 3, value=dolfin_mesh.domains())
    submesh_11  = d.MeshView.create(mf_3, 11)
    submesh_12  = d.MeshView.create(mf_3, 12)

    V11         = d.FunctionSpace(submesh_11, "CG", 1)
    V12         = d.FunctionSpace(submesh_12, "CG", 1)
    V           = d.MixedFunctionSpace(V11, V12)
    u11, u12    = d.TrialFunctions(V)
    v11, v12    = d.TestFunctions(V)
    F           = d.inner(d.grad(u11), d.grad(v11))

#todo:
# Move repeated things to fixtures (e.g. mf_3)
# Test: time-independent two subvolume problem (with coupling)
# Test: time-dependent two subvolume problem (with coupling) [e.g. leakage with constant flux, assert mass at each time]
# Test: ^ same as above but using stubs
# Test: does monolithic solver work in parallel? (check size of matrices on each cpu, check time)

