import pytest

import stubs


def test_Compartment_initialization(compartment_kwargs_Cyto):
    Cyto = stubs.model_assembly.Compartment(**compartment_kwargs_Cyto)

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
    Cyto = stubs.model_assembly.Compartment(**compartment_kwargs_Cyto)
    Cyto.dolfin_mesh


@pytest.mark.xfail
def test_Compartment_access_mesh_id(compartment_kwargs_Cyto):
    Cyto = stubs.model_assembly.Compartment(**compartment_kwargs_Cyto)
    Cyto.mesh_id


@pytest.mark.xfail
def test_Compartment_access_num_cells(compartment_kwargs_Cyto):
    Cyto = stubs.model_assembly.Compartment(**compartment_kwargs_Cyto)
    Cyto.num_cells


@pytest.mark.xfail
def test_Compartment_access_num_facets(compartment_kwargs_Cyto):
    Cyto = stubs.model_assembly.Compartment(**compartment_kwargs_Cyto)
    Cyto.num_facets


@pytest.mark.xfail
def test_Compartment_access_num_vertices(compartment_kwargs_Cyto):
    Cyto = stubs.model_assembly.Compartment(**compartment_kwargs_Cyto)
    Cyto.num_vertices


@pytest.mark.xfail
def test_Compartment_access_nvolume(compartment_kwargs_Cyto):
    Cyto = stubs.model_assembly.Compartment(**compartment_kwargs_Cyto)
    Cyto.nvolume
