import math
import numpy as np
import pytest
import sympy as sym

import smart


def test_Parameter_initialization(parameter_kwargs_k3f):
    """Test that we can initialize k3f SpeciesContainer"""
    k3f = smart.model_assembly.Parameter(**parameter_kwargs_k3f)

    assert k3f.name == parameter_kwargs_k3f["name"]
    assert math.isclose(k3f.value, parameter_kwargs_k3f["value"])
    assert np.allclose(k3f.value_vector, [0.0, parameter_kwargs_k3f["value"]])

    assert k3f.unit == parameter_kwargs_k3f["unit"]
    assert k3f.quantity == parameter_kwargs_k3f["value"] * parameter_kwargs_k3f["unit"]

    assert k3f.type == "constant"
    assert k3f.group == parameter_kwargs_k3f["group"]
    assert k3f.use_preintegration is parameter_kwargs_k3f["use_preintegration"]
    assert k3f.is_space_dependent is False
    assert k3f.is_time_dependent is False
    assert k3f.notes == parameter_kwargs_k3f["notes"]


@pytest.mark.parametrize("use_preintegration", [True, False])
def test_Parameter_from_expression(use_preintegration):
    Vmax, t0, m = 500, 0.1, 200
    t = sym.symbols("t")
    pulseI = Vmax * sym.atan(m * (t - t0))
    pulse = sym.diff(pulseI, t)
    value = Vmax * m / (1 + (m * (0.0 - t0)) ** 2)

    flux_unit = smart.units.unit.molecule / (smart.units.unit.um**2 * smart.units.unit.sec)
    j1pulse = smart.model_assembly.Parameter.from_expression(
        "j1pulse",
        pulse,
        flux_unit,
        use_preintegration=use_preintegration,
        preint_sym_expr=pulseI,
    )

    assert math.isclose(j1pulse.value, value)
    assert np.allclose(j1pulse.value_vector, [0.0, value])

    assert j1pulse.unit == smart.units.unit_to_quantity(flux_unit)
    assert np.isclose(j1pulse.quantity, value * flux_unit)

    assert j1pulse.type == "expression"
    assert j1pulse.use_preintegration is use_preintegration
    assert j1pulse.is_space_dependent is False
    assert j1pulse.is_time_dependent is True


def test_ParameterContainer(parameter_kwargs_k3f):
    """Test that we can initialize k3f SpeciesContainer"""
    k3f = smart.model_assembly.Parameter(**parameter_kwargs_k3f)
    k3r = smart.model_assembly.Parameter("k3r", 100, 1 / smart.units.unit.sec)
    pc = smart.model_assembly.ParameterContainer()
    assert pc.size == 0
    pc.add([k3f])
    assert pc.size == 1
    pc["k3f"] == k3f
    pc.add([k3f])
    # Adding same Parameter should not do anything
    assert pc.size == 1
    pc.add([k3r])
    assert pc.size == 2
    assert set(pc.keys) == {"k3f", "k3r"}
