import stubs
import pytest

"""
Test to see if stubs can write a sbmodel
"""
@pytest.mark.stubs
def test_stubs_define_sbmodel():
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
    assert type(p_in) == stubs.model_building.ParameterDF
    assert type(s_in) == stubs.model_building.SpeciesDF
    assert type(c_in) == stubs.model_building.CompartmentDF
    assert type(r_in) == stubs.model_building.ReactionDF

    assert p_in.df.shape[0] == s_in.df.shape[0] == c_in.df.shape[0] == r_in.df.shape[0] == 1
