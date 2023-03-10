import math
import numpy as np
import pytest
import sympy as sym

import stubs


def test_Parameter_initialization(parameter_kwargs_k3f):
    """Test that we can initialize k3f SpeciesContainer"""
    k3f = stubs.model_assembly.Parameter(**parameter_kwargs_k3f)

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


@pytest.mark.xfail
def test_Parameter_from_file():
    # Need to figure out how the file should look like
    raise NotImplementedError


@pytest.mark.parametrize("use_preintegration", [True, False])
def test_Parameter_from_expression(use_preintegration):
    Vmax, t0, m = 500, 0.1, 200
    t = sym.symbols("t")
    pulseI = Vmax * sym.atan(m * (t - t0))
    pulse = sym.diff(pulseI, t)
    value = Vmax * m / (1 + (m * (0.0 - t0)) ** 2)

    flux_unit = stubs.unit.molecule / (stubs.unit.um**2 * stubs.unit.sec)
    j1pulse = stubs.model_assembly.Parameter.from_expression(
        "j1pulse",
        pulse,
        flux_unit,
        use_preintegration=use_preintegration,
        preint_sym_expr=pulseI,
    )

    assert math.isclose(j1pulse.value, value)
    assert np.allclose(j1pulse.value_vector, [0.0, value])

    assert j1pulse.unit == stubs.common.pint_unit_to_quantity(flux_unit)
    assert np.isclose(j1pulse.quantity, value * flux_unit)

    assert j1pulse.type == "expression"
    assert j1pulse.use_preintegration is use_preintegration
    assert j1pulse.is_space_dependent is False
    assert j1pulse.is_time_dependent is True


@pytest.mark.xfail
def test_access_dolfin_quatity(parameter_kwargs_k3f):
    k3f = stubs.model_assembly.Parameter(**parameter_kwargs_k3f)
    # We should have proper error handling here
    k3f.dolfin_quantity


def test_ParameterContainer(parameter_kwargs_k3f):
    """Test that we can initialize k3f SpeciesContainer"""
    k3f = stubs.model_assembly.Parameter(**parameter_kwargs_k3f)
    k3r = stubs.model_assembly.Parameter("k3r", 100, 1 / stubs.unit.sec)
    pc = stubs.model_assembly.ParameterContainer()
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
