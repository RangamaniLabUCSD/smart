#import stubs
#import pytest
#
## Fixtures
#@pytest.fixture
#def stubs_mesh(mesh_filename):
#    return stubs.mesh.ParentMesh(mesh_filename=mesh_filename)
#
#@pytest.fixture
#def stubs_config():
#    return stubs.config.Config()
#
## Tests
#@pytest.mark.stubs_model_init
#def test_stubs_mesh_load_dolfin_mesh(stubs_mesh):
#    "Make sure that stubs is loading the dolfin mesh when we create a ParentMesh"
#    assert stubs_mesh.dolfin_mesh.num_vertices() > 1
#    assert stubs_mesh.dolfin_mesh.num_cells() > 1