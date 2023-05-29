import pytest

import smart


def test_Compartment_initialization(compartment_kwargs_Cyto):
    """Test that we can initialize a Compartment"""
    Cyto = smart.model_assembly.Compartment(**compartment_kwargs_Cyto)

    assert Cyto.V is None
    assert Cyto.cell_marker == compartment_kwargs_Cyto["cell_marker"]
    assert Cyto.dimensionality == compartment_kwargs_Cyto["dimensionality"]
    assert Cyto.name == compartment_kwargs_Cyto["name"]
    assert Cyto.num_dofs == 0
    assert Cyto.num_dofs_local == 0
    assert Cyto.species == {}
    assert Cyto.u == {}
    assert Cyto.v is None


@pytest.mark.xfail
def test_Compartment_access_dolfin_mesh(compartment_kwargs_Cyto):
    Cyto = smart.model_assembly.Compartment(**compartment_kwargs_Cyto)
    Cyto.dolfin_mesh


@pytest.mark.xfail
def test_Compartment_access_mesh_id(compartment_kwargs_Cyto):
    Cyto = smart.model_assembly.Compartment(**compartment_kwargs_Cyto)
    Cyto.mesh_id


@pytest.mark.xfail
def test_Compartment_access_num_cells(compartment_kwargs_Cyto):
    Cyto = smart.model_assembly.Compartment(**compartment_kwargs_Cyto)
    Cyto.num_cells


@pytest.mark.xfail
def test_Compartment_access_num_facets(compartment_kwargs_Cyto):
    Cyto = smart.model_assembly.Compartment(**compartment_kwargs_Cyto)
    Cyto.num_facets


@pytest.mark.xfail
def test_Compartment_access_num_vertices(compartment_kwargs_Cyto):
    Cyto = smart.model_assembly.Compartment(**compartment_kwargs_Cyto)
    Cyto.num_vertices


@pytest.mark.xfail
def test_Compartment_access_nvolume(compartment_kwargs_Cyto):
    Cyto = smart.model_assembly.Compartment(**compartment_kwargs_Cyto)
    Cyto.nvolume


def test_CompartmentContainer(compartment_kwargs_Cyto, compartment_kwargs_PM):
    """Test that we can initialize a CompartmentContainer"""
    Cyto = smart.model_assembly.Compartment(**compartment_kwargs_Cyto)
    PM = smart.model_assembly.Compartment(**compartment_kwargs_PM)
    cc = smart.model_assembly.CompartmentContainer()
    assert cc.size == 0
    cc.add([Cyto])
    assert cc.size == 1
    cc["Cyto"] == Cyto
    cc.add([Cyto])
    # Adding same species should not do anything
    assert cc.size == 1
    cc.add([PM])
    assert cc.size == 2
    assert set(cc.keys) == {"Cyto", "PM"}
