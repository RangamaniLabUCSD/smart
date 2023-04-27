import math

import pytest

import stubs


@pytest.fixture(name="A_kwargs")
def example_species():
    kwargs = dict(
        concentration_units=stubs.unit.uM,
        diffusion_units=stubs.unit.um**2 / stubs.unit.sec,
        initial_condition=0.01,
        D=2.0,
        name="A",
        compartment_name="Cyto",
        group="Some group",
    )

    A = stubs.model_assembly.Species(**kwargs)
    return (A, kwargs)


def test_Species_initialization(A_kwargs):
    A, kwargs = A_kwargs
    assert A.name == kwargs["name"]
    assert A.latex_name == kwargs["name"]
    assert str(A.sym) == kwargs["name"]

    assert math.isclose(A.initial_condition, kwargs["initial_condition"])
    assert (
        A.initial_condition_quantity == kwargs["initial_condition"] * kwargs["concentration_units"]
    )
    assert A.concentration_units == stubs.common.pint_unit_to_quantity(
        kwargs["concentration_units"]
    )
    assert math.isclose(A.D, kwargs["D"])
    assert A.diffusion_units == stubs.common.pint_unit_to_quantity(kwargs["diffusion_units"])
    assert A.D_quantity == kwargs["D"] * kwargs["diffusion_units"]

    assert A.compartment_name == kwargs["compartment_name"]
    assert A.group == kwargs["group"]
    assert A.dof_map is None
    assert A.ut is None
    assert A.v is None
    assert A.u == {}
    assert A.is_an_added_species is False
    assert A.is_in_a_reaction is False


@pytest.mark.xfail
def test_access_vscalar(A_kwargs):
    A, kwargs = A_kwargs
    # We should have proper error handling here
    A.vscalar


@pytest.mark.xfail
def test_access_dolfin_quatity(A_kwargs):
    A, kwargs = A_kwargs
    # We should have proper error handling here
    A.dolfin_quantity
