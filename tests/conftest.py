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
def smart_config(request):
    return smart.config.Config()
