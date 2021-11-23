import stubs
import pytest

# Fixtures
@pytest.fixture
def mesh_filename(datadir):
    return str(datadir.joinpath('adjacent_cubes.xml'))

@pytest.fixture
def stubs_mesh(mesh_filename):
    return stubs.mesh.ParentMesh(mesh_filename=mesh_filename)

@pytest.fixture
def stubs_config():
    return stubs.config.Config()

@pytest.fixture
def stubs_config():
    return stubs.config.Config()

# Tests
@pytest.mark.stubs
def test_stubs_mesh_load_dolfin_mesh(stubs_mesh):
    "Make sure that stubs is loading the dolfin mesh when we create a ParentMesh"
    assert stubs_mesh.dolfin_mesh.num_vertices() > 1
    assert stubs_mesh.dolfin_mesh.num_cells() > 1

@pytest.mark.stubs
def test_stubs_define_sbmodel():
    "Test to see if stubs can write a sbmodel"
    unit = stubs.unit # unit registry

    # initialize 
    p, s, c, r = stubs.model_building.empty_sbmodel()

    ## define parameters
    # name, value, unit, notes
    p.append('kdeg', 5.0, 1/(unit.s), 'degradation rate')

    ## define species
    # name, plot group, concentration units, initial condition, diffusion
    # coefficient, diffusion coefficient units, compartment
    s.append('B', 'cytosolic', unit.uM, 10, 1, unit.um**2/unit.s, 'cyto')

    ## define compartments
    # name, geometric dimensionality, length scale units, marker value
    c.append('cyto', 3, unit.um, 1)

    ## define reactions
    # name, notes, left hand side of reaction, right hand side of reaction, kinetic
    # parameters
    r.append('B linear degredation', 'example reaction', ['B'], [], {"on": "kdeg"}, reaction_type='mass_action_forward')

    assert p.df.shape[0] == s.df.shape[0] == c.df.shape[0] == r.df.shape[0] == 1

    # write out to file
    stubs.common.write_sbmodel('pytest.sbmodel', p, s, c, r)

    # read in file
    p_in, s_in, c_in, r_in = stubs.common.read_sbmodel('pytest.sbmodel', output_type=tuple)
    assert type(p_in) == stubs.model_assembly.ParameterContainer
    assert type(s_in) == stubs.model_assembly.SpeciesContainer
    assert type(c_in) == stubs.model_assembly.CompartmentContainer
    assert type(r_in) == stubs.model_assembly.ReactionContainer

    assert p.df.shape[0] == s.df.shape[0] == c.df.shape[0] == r.df.shape[0] == 1
    assert p_in.size == s_in.size == c_in.size == r_in.size == 1

@pytest.mark.stubs
def test_stubs_load_mesh(stubs_mesh):
    "Test that stubs is loading the dolfin mesh when we create a ParentMesh"
    assert stubs_mesh.dolfin_mesh.num_vertices() > 1
    assert stubs_mesh.dolfin_mesh.num_cells() > 1
    
@pytest.mark.stubs
def test_stubs_define_model():
    "Test that stubs can generate a config file"
    config = stubs.config.Config()