import pytest
import stubs
from pathlib import Path

@pytest.fixture(scope='module')
def rootdir(request):
    # Start at the directory of request and work backwards until we reach '/.../.../stubs/tests'
    path = Path('.').resolve()
    while True:
        if path.parts[-1]=='stubs' and path.joinpath('tests').is_dir():
            return path
        path = path.parent

@pytest.fixture(scope='module')
def testdir(request):
    # Start at the directory of request and work backwards until we reach '/.../.../stubs/tests'
    path    = Path('.').resolve()
    subdir  = 'tests'
    while True:
        if path.parts[-1]=='stubs' and path.joinpath(subdir).is_dir():
            return path.joinpath(subdir)
        path = path.parent

@pytest.fixture(scope='module')
def datadir(request):
    # Start at the directory of request and work backwards until we reach '/.../.../stubs/data'
    path    = Path('.').resolve()
    subdir  = 'data'
    while True:
        if path.parts[-1]=='stubs' and path.joinpath(subdir).is_dir():
            return path.joinpath(subdir)
        path = path.parent

@pytest.fixture(scope='module')
def mesh_filename(datadir):
    return str(datadir.joinpath('adjacent_cubes.xml'))

@pytest.fixture(scope='module')
def stubs_mesh(mesh_filename):
    return stubs.mesh.ParentMesh(mesh_filename=mesh_filename)

@pytest.fixture(scope='module')
def stubs_config(request):
    return stubs.config.Config()

