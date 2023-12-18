import dolfin
from smart import mesh_tools


def test_save_load_mesh(tmp_path):
    mesh = dolfin.UnitCubeMesh(3, 3, 3)
    mf_cell = dolfin.MeshFunction("size_t", mesh, 3)
    mf_cell.array()[:] = 1
    mf_facet = dolfin.MeshFunction("size_t", mesh, 2)
    mf_facet.array()[:] = 2

    mesh_tools.write_mesh(
        mesh=mesh,
        mf_facet=mf_facet,
        mf_cell=mf_cell,
        filename=tmp_path.with_suffix(".h5"),
    )

    geo = mesh_tools.load_mesh(filename=tmp_path.with_suffix(".h5"))
    assert mesh is not geo.mesh  # Different mesh
    assert (mesh.coordinates() == geo.mesh.coordinates()).all()
    assert (mf_cell.array() == geo.mf_cell.array()).all()
    assert (mf_facet.array() == geo.mf_facet.array()).all()
    assert geo.filename == tmp_path.with_suffix(".h5")


def test_save_load_mesh_with_existing_mesh(tmp_path):
    mesh = dolfin.UnitCubeMesh(3, 3, 3)
    mf_cell = dolfin.MeshFunction("size_t", mesh, 3)
    mf_cell.array()[:] = 1
    mf_facet = dolfin.MeshFunction("size_t", mesh, 2)
    mf_facet.array()[:] = 2

    mesh_tools.write_mesh(
        mesh=mesh,
        mf_facet=mf_facet,
        mf_cell=mf_cell,
        filename=tmp_path.with_suffix(".h5"),
    )

    geo = mesh_tools.load_mesh(filename=tmp_path.with_suffix(".h5"), mesh=mesh)
    assert mesh is geo.mesh  # Same mesh
    assert (mf_cell.array() == geo.mf_cell.array()).all()
    assert (mf_facet.array() == geo.mf_facet.array()).all()
    assert geo.filename == tmp_path.with_suffix(".h5")
