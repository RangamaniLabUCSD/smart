# Classes/functions used to construct models
from stubs import unit

import pandas as pd 
import dolfin as d
import pint 

# conversion factors
def ucyto_to_upm(value, cyto2PM):
    # convert a volume concentration in the cyto to a concentration on the pm (1/um**2)
    return (value * cyto2PM).to(unit.molecule/unit.um**2)
def ucyto_to_uer(value, cyto2ER):
    # convert a volume concentration in the cyto to a concentration on the er
    return (value * cyto2ER).to(unit.molecule/unit.um**2)
#def jpm_to_bflux(value):
#    # convert a surface flux (eg 1/(um^2*s)) to a boundary flux
#    return value / unit.avogadro_number
def kfcyto_to_kfpm(value, cyto2PM):
    # convert forward reaction rate in cytosol into one on the PM
    # e.g. 1/(uM*s) -> um**2/(molecule * s)
    return (value / cyto2PM).to(unit.um**2/(unit.molecule * unit.s))

class ParameterDF(object):
    def __init__(self):
        #self.df = pd.DataFrame(columns=["value", "unit", "group", "notes", "is_time_dependent", "dolfinConstant", "symExpr"])
        self.df = pd.DataFrame({'value': pd.Series([], dtype=float),
                                'unit': pd.Series([], dtype=object),
                                'group': pd.Series([], dtype=str),
                                'notes': pd.Series([], dtype=str),
                                'is_time_dependent': pd.Series([], dtype=bool),
                                'is_preintegrated': pd.Series([], dtype=bool),
                                'dolfinConstant': pd.Series([], dtype=object),
                                'symExpr': pd.Series([], dtype=object)})

    def append(self, name, value, unit, group, notes='', is_time_dependent=False, is_preintegrated=False, dolfinConstant=None, symExpr=None):
        if is_time_dependent:
            if not symExpr:
                raise Exception("A time-dependent parameter must have a sympy expression!")

        self.df = self.df.append(pd.Series({"value": value, "unit": str(unit),
                                            "group": group, "notes": notes,
                                            "is_time_dependent": is_time_dependent,
                                            "is_preintegrated": is_preintegrated,
                                            "dolfinConstant": dolfinConstant,
                                            "symExpr": symExpr}, name=name))#, ignore_index=True)

    def write_json(self, name='parameters.json'):
        self.df.to_json(name)


class SpeciesDF(object):
    """
    IC assumed to be in terms concentration units
    """
    def __init__(self):
        self.df = pd.DataFrame({'initial_condition': pd.Series([], dtype=float),
                                'concentration_units': pd.Series([], dtype=object),
                                'D': pd.Series([], dtype=float),
                                'D_units': pd.Series([], dtype=object),
                                'compartment_name': pd.Series([], dtype=str),
                                'group': pd.Series([], dtype=str)})

    def append(self, name, group, units, IC, D, D_units, compartment_name):
        self.df = self.df.append(pd.Series({"group": group, "concentration_units": str(units),
                                            "initial_condition": IC, "D": D, "D_units": str(D_units),
                                            "compartment_name": compartment_name}, name=name))#, ignore_index=True)

    def write_json(self, name='species.json'):
        self.df.to_json(name)

class CompartmentDF(object):
    def __init__(self):
        self.df = pd.DataFrame({'dimensionality': pd.Series([], dtype=int),
                                'compartment_units': pd.Series([], dtype=object),
                                'cell_marker': pd.Series([], dtype=int)})

    def append(self, name, dim, units, marker):
        self.df = self.df.append(pd.Series({"dimensionality": dim, "compartment_units": str(units),
                                            "cell_marker": marker}, name=name))#, ignore_index=True)

    def write_json(self, name='compartments.json'):
        self.df.to_json(name)


class ReactionDF(object):
    def __init__(self):
        self.df = pd.DataFrame(columns=["group", "LHS", "RHS",
                                        "paramDict", "reaction_type",
                                        "explicit_restriction_to_domain", "speciesDict"])
        self.df = pd.DataFrame({'group': pd.Series([], dtype=str),
                                'LHS': pd.Series([], dtype=object),
                                'RHS': pd.Series([], dtype=object),
                                'paramDict': pd.Series([], dtype=object),
                                'reaction_type': pd.Series([], dtype=str),
                                'explicit_restriction_to_domain': pd.Series([], dtype=object),
                                'speciesDict': pd.Series([], dtype=object)})
    def append(self, name, group, LHS, RHS, paramDict,
               reaction_type="mass_action", explicit_restriction_to_domain=None, speciesDict={}):

        self.df = self.df.append(pd.Series({"group": group, "LHS": LHS,
                                  "RHS": RHS, "paramDict": paramDict,
                                  "reaction_type": reaction_type,
                                  "explicit_restriction_to_domain": explicit_restriction_to_domain,
                                  "speciesDict":speciesDict}, name=name))#, ignore_index=True)

    def write_json(self, name='reactions.json'):
        self.df.to_json(name)




