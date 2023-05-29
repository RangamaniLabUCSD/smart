import pytest
import smart
from pathlib import Path
import os


@pytest.fixture(scope="module")
def datadir(request):
    """
    Use the path of the requesting file (assumed to be within the tests folder)
    to find the data directory
    """
    subdir = "data"
    test_dir = os.path.dirname(os.path.abspath(request.module.__file__))
    return Path(test_dir).joinpath(f"../{subdir}")


@pytest.fixture(scope="module")
def mesh_filename(datadir):
    return str(datadir.joinpath("adjacent_cubes.xml"))


@pytest.fixture(scope="module")
def smart_mesh(mesh_filename):
    return smart.mesh.ParentMesh(
        mesh_filename=mesh_filename,
        mesh_filetype=mesh_filename.split(".")[-1],
        name="test_mesh",
    )


@pytest.fixture(scope="module")
def stubs_config(request):
    return smart.config.Config()


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
