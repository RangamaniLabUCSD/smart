"""Pint registry from SMART and related convenience function"""
__all__ = ["unit", "unit_to_quantity", "quantity_to_unit"]

import pint

unit = pint.UnitRegistry()
unit.define("molecule = mol/6.022140857e23")
unit.define("nM_10 = 10*nM")
unit.define("nM_100 = 100*nM")
unit.define("uM_10 = 10*uM")
unit.define("uM_100 = 100*uM")
unit.define("molec_10_per_um2 = 10*molecule/um**2")
unit.define("molec_100_per_um2 = 100*molecule/um**2")
unit.define("molec_1000_per_um2 = 1000*molecule/um**2")
unit.define("molec_div10_per_um2 = 1/10*molecule/um**2")
unit.define("molec_div100_per_um2 = 1/100*molecule/um**2")
unit.define("molec_div1000_per_um2 = 1/1000*molecule/um**2")


def unit_to_quantity(pint_unit: pint.Unit):
    """
    Convert a `pint.Unit` to a `pint.Quantity`
    """
    if not isinstance(pint_unit, pint.Unit):
        raise TypeError("Input must be a pint unit")
    # Use unit registry as `pint.Quantity` would create a new registry
    return unit.Quantity(1, pint_unit)


def quantity_to_unit(pint_quantity: pint.Quantity):
    """
    Get the unit of a `pint.Quantity` (has to have magnitude one).
    """
    if not isinstance(pint_quantity, pint.Quantity):
        raise TypeError("Input must be a pint quantity")
    if pint_quantity.magnitude != 1.0:
        raise ValueError("Trying to convert a pint quantity into a unit with magnitude != 1")
    return pint_quantity.units
