import pytest
import smart


@pytest.fixture(scope="session")
def species_kwargs_A():
    return dict(
        concentration_units=smart.units.unit.uM,
        diffusion_units=smart.units.unit.um**2 / smart.units.unit.sec,
        initial_condition=0.01,
        D=2.0,
        name="A",
        compartment_name="Cyto",
        group="Some group",
    )


@pytest.fixture(scope="session")
def species_kwargs_AER():
    return dict(
        concentration_units=smart.units.unit.uM,
        diffusion_units=smart.units.unit.um**2 / smart.units.unit.sec,
        initial_condition=200.0,
        D=5.0,
        name="AER",
        compartment_name="ER",
        group="Another group",
    )


@pytest.fixture
def compartment_kwargs_Cyto():
    return dict(
        dimensionality=3,
        name="Cyto",
        compartment_units=smart.units.unit.um,
        cell_marker=1,
    )


@pytest.fixture
def compartment_kwargs_PM():
    return dict(
        dimensionality=2,
        name="PM",
        compartment_units=smart.units.unit.um,
        cell_marker=10,
    )


@pytest.fixture
def parameter_kwargs_k3f():
    return dict(
        name="k3f",
        value=100,
        unit=1 / (smart.units.unit.uM * smart.units.unit.sec),
        group="some group",
        notes="Some notes",
        use_preintegration=True,
    )
