import smart


def test_Compartment_initialization(compartment_kwargs_Cyto):
    """Test that we can initialize a Compartment"""
    Cyto = smart.model_assembly.Compartment(**compartment_kwargs_Cyto)

    assert Cyto.V is None
    assert Cyto.cell_marker == compartment_kwargs_Cyto["cell_marker"]
    assert Cyto.dimensionality == compartment_kwargs_Cyto["dimensionality"]
    assert Cyto.name == compartment_kwargs_Cyto["name"]
    assert Cyto.num_dofs == 0
    assert Cyto.num_dofs_local == 0
    assert Cyto.species == {}
    assert Cyto.u == {}
    assert Cyto.v is None


def test_CompartmentContainer(compartment_kwargs_Cyto, compartment_kwargs_PM):
    """Test that we can initialize a CompartmentContainer"""
    Cyto = smart.model_assembly.Compartment(**compartment_kwargs_Cyto)
    PM = smart.model_assembly.Compartment(**compartment_kwargs_PM)
    cc = smart.model_assembly.CompartmentContainer()
    assert cc.size == 0
    cc.add([Cyto])
    assert cc.size == 1
    cc["Cyto"] == Cyto
    cc.add([Cyto])
    # Adding same species should not do anything
    assert cc.size == 1
    cc.add([PM])
    assert cc.size == 2
    assert set(cc.keys) == {"Cyto", "PM"}
