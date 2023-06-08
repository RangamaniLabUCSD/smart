import math

import pytest

import smart


def test_Species_initialization(species_kwargs_A):
    """Test that we can initialize a SpeciesContainer"""
    A = smart.model_assembly.Species(**species_kwargs_A)
    assert A.name == species_kwargs_A["name"]
    assert A.latex_name == species_kwargs_A["name"]
    assert str(A.sym) == species_kwargs_A["name"]

    assert math.isclose(A.initial_condition, species_kwargs_A["initial_condition"])
    assert (
        A.initial_condition_quantity
        == species_kwargs_A["initial_condition"] * species_kwargs_A["concentration_units"]
    )
    assert A.concentration_units == smart.units.unit_to_quantity(
        species_kwargs_A["concentration_units"]
    )
    assert math.isclose(A.D, species_kwargs_A["D"])
    assert A.diffusion_units == smart.units.unit_to_quantity(species_kwargs_A["diffusion_units"])
    assert A.D_quantity == species_kwargs_A["D"] * species_kwargs_A["diffusion_units"]

    assert A.compartment_name == species_kwargs_A["compartment_name"]
    assert A.group == species_kwargs_A["group"]
    assert A.dof_map is None
    assert A.ut is None
    assert A.v is None
    assert A.u == {}
    assert A.is_an_added_species is False
    assert A.is_in_a_reaction is False


def test_SpeciesContainer(species_kwargs_A, species_kwargs_AER):
    """Test that we can initialize a SpeciesContainer"""
    A = smart.model_assembly.Species(**species_kwargs_A)
    AER = smart.model_assembly.Species(**species_kwargs_AER)
    sc = smart.model_assembly.SpeciesContainer()
    assert sc.size == 0
    sc.add([A])
    assert sc.size == 1
    sc["A"] == A
    sc.add([A])
    # Adding same species should not do anything
    assert sc.size == 1
    sc.add([AER])
    assert sc.size == 2
    assert set(sc.keys) == {"A", "AER"}


def test_add_non_Species_to_SpeciesContainer_raises_InvalidObjectException(
    compartment_kwargs_Cyto,
):
    """Test that if we try to add something different than a species
    to a SpeciesContainer then an exception will be raised"""
    sc = smart.model_assembly.SpeciesContainer()
    Cyto = smart.model_assembly.Compartment(**compartment_kwargs_Cyto)
    with pytest.raises(smart.model_assembly.InvalidObjectException):
        sc.add([Cyto])
