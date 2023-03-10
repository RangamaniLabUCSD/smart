import pytest
import stubs
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
def stubs_mesh(mesh_filename):
    return stubs.mesh.ParentMesh(
        mesh_filename=mesh_filename,
        mesh_filetype=mesh_filename.split(".")[-1],
        name="test_mesh",
    )


@pytest.fixture(scope="module")
def stubs_config(request):
    return stubs.config.Config()


@pytest.fixture(scope="session")
def species_kwargs_A():
    return dict(
        concentration_units=stubs.unit.uM,
        diffusion_units=stubs.unit.um**2 / stubs.unit.sec,
        initial_condition=0.01,
        D=2.0,
        name="A",
        compartment_name="Cyto",
        group="Some group",
    )


@pytest.fixture(scope="session")
def species_kwargs_AER():
    return dict(
        concentration_units=stubs.unit.uM,
        diffusion_units=stubs.unit.um**2 / stubs.unit.sec,
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
        compartment_units=stubs.unit.um,
        cell_marker=1,
    )


@pytest.fixture
def compartment_kwargs_PM():
    return dict(
        dimensionality=2,
        name="PM",
        compartment_units=stubs.unit.um,
        cell_marker=10,
    )
