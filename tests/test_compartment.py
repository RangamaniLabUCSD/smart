import pytest

import stubs


@pytest.fixture(name="Cyto_kwargs")
def example_compartment():
    kwargs = dict(
        dimensionality=3,
        name="Cyto",
        compartment_units=stubs.unit.um,
        cell_marker=1,
    )

    Cyto = stubs.model_assembly.Compartment(**kwargs)
    return (Cyto, kwargs)


def test_Compartment_initialization(Cyto_kwargs):
    Cyto, kwargs = Cyto_kwargs

    assert Cyto.V is None
    assert Cyto.cell_marker == kwargs["cell_marker"]
    assert Cyto.dimensionality == kwargs["dimensionality"]
    assert Cyto.name == kwargs["name"]
    assert Cyto.num_dofs == 0
    assert Cyto.num_dofs_local == 0
    assert Cyto.species == {}
    assert Cyto.u == {}
    assert Cyto.v is None


@pytest.mark.xfail
def test_Compartment_access_dolfin_mesh(Cyto_kwargs):
    Cyto, kwargs = Cyto_kwargs
    Cyto.dolfin_mesh


@pytest.mark.xfail
def test_Compartment_access_mesh_id(Cyto_kwargs):
    Cyto, kwargs = Cyto_kwargs
    Cyto.mesh_id


@pytest.mark.xfail
def test_Compartment_access_num_cells(Cyto_kwargs):
    Cyto, kwargs = Cyto_kwargs
    Cyto.num_cells


@pytest.mark.xfail
def test_Compartment_access_num_facets(Cyto_kwargs):
    Cyto, kwargs = Cyto_kwargs
    Cyto.num_facets


@pytest.mark.xfail
def test_Compartment_access_num_vertices(Cyto_kwargs):
    Cyto, kwargs = Cyto_kwargs
    Cyto.num_vertices


@pytest.mark.xfail
def test_Compartment_access_nvolume(Cyto_kwargs):
    Cyto, kwargs = Cyto_kwargs
    Cyto.nvolume
