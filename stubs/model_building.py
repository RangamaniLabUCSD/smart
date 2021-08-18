"""Classes/functions used to construct models
"""
from stubs import unit

import pandas as pd 
import dolfin as d
import pint 

# IN PROGRESS
# class ObjectDF(object):
#     def __init__(self, property_types):
#         """General dataframe wrapper class which can hold parameters, species, etc.
        
#         Args:
#             property_types (dict): property_name -> data type. Each key/value
#             pair will correspond to a pandas Series.
#         """
#         self.property_types
#         data_dict = dict()
#         for property_name, property_type in property_types.items():
#             data_dict.update({property_name, 
#                              pd.Series([], dtype=property_type)})
#         self.df = pd.DataFrame(data_dict)

#         def add(self, name, **kwargs):
#             inputs = locals()
#             to_df = kwargs.update({"name": name})
#             # ensure types are correct
#             for key, value in to_df.items():
#                 to_df[key] =  

class ParameterDF(object):
    """
    A standard (non time-dependent) parameter has an associated value, unit,
    group [i.e. what physics does the parameter belong to?], notes, and a 
    dolfinConstant [a dolfin Constant class with value being adjusted for unit
    consistency and/or time-dependency].
    A time-dependent parameter can either be expressed as a symbolic expression
    wrt t or it can be sampled from some dataset (a .csv file with time as the
    first column and values as the second column). If the parameter is provided
    as a symbolic expression it can also be provided in pre-integrated format,
    an expression which is defined as 

    \f$p_I(t) = \int_0^t p(x) \,dx \f$

    This is useful for enabling larger time-steps when using functions with
    rapidly changing values.
    """
    def __init__(self):
        self.df = pd.DataFrame({'value': pd.Series([], dtype=float),
                                'unit': pd.Series([], dtype=object),
                                'group': pd.Series([], dtype=str),
                                'notes': pd.Series([], dtype=str),
                                'is_time_dependent': pd.Series([], dtype=bool),
                                # is the data sampled from some discrete data?
                                'sampling_file': pd.Series([], dtype=str), 
                                'sampling_data': pd.Series([], dtype=object), 
                                'preint_sampling_data': pd.Series([], dtype=object), 
                                'dolfinConstant': pd.Series([], dtype=object),
                                'symExpr': pd.Series([], dtype=object),
                                'preintegrated_symExpr': pd.Series([], dtype=object)})

    def append(self, name, value, unit, group, notes='', is_time_dependent=False,
               sampling_file='', sampling_data=None, dolfinConstant=None,
               symExpr=None, preint_sampling_data=None, preintegrated_symExpr=None):
        """
        Adds data to the parameter
        """

        # check that the parameter is not under or over specified
        if is_time_dependent:
            if symExpr and sampling_file:
                raise Exception("Provide parameter %s with either a symbolic"\
                                "expression or discrete data, not both!" % name)
            if not (symExpr or sampling_file):
                raise Exception("Time-dependent parameter %s must either have"\
                                "a sympy expression or have discrete data!"\
                                % name)

        self.df = self.df.append(pd.Series({"value": value, "unit": str(unit),
                                            "group": group, "notes": notes,
                                            "is_time_dependent": is_time_dependent,
                                            # is the data sampled from some discrete data?
                                            'sampling_file': sampling_file,
                                            'sampling_data': sampling_data,
                                            'preint_sampling_data': preint_sampling_data,
                                            "dolfinConstant": dolfinConstant,
                                            "symExpr": symExpr,
                                            "preintegrated_symExpr": preintegrated_symExpr},
                                            name=name))

    def write_json(self, name='parameters.json'):
        self.df.to_json(name)
        print("Parameters generated successfully! Saved as %s" % name)



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
        self.df = self.df.append(pd.Series({"group": group,
                                            "concentration_units": str(units),
                                            "initial_condition": IC, "D": D,
                                            "D_units": str(D_units),
                                            "compartment_name": compartment_name}, name=name))

    def write_json(self, name='species.json'):
        self.df.to_json(name)
        print("Species generated successfully! Saved as %s" % name)

class CompartmentDF(object):
    def __init__(self):
        self.df = pd.DataFrame({'dimensionality': pd.Series([], dtype=int),
                                'compartment_units': pd.Series([], dtype=object),
                                'cell_marker': pd.Series([], dtype=object)})

    def append(self, name, dim, units, marker):
        self.df = self.df.append(pd.Series({"dimensionality": dim,
                                           "compartment_units": str(units),
                                            "cell_marker": marker}, name=name))

    def write_json(self, name='compartments.json'):
        self.df.to_json(name)
        print("Compartment generated successfully! Saved as %s" % name)


class ReactionDF(object):
    """
    A reaction is specified by the reactants/products on the left hand side 
    (LHS), right hand side (RHS), kinetic parameters, the reaction type (STUBS
    will always assume mass action unless specified otherwise).

    If a reaction contains species from two different compartments
    then STUBS will automatically apply boundary conditions and scale the fluxes
    appropriately. However sometimes a flux is dependent only on a species in
    one compartment yet is still a boundary flux -
    explicit_restriction_to_domain allows the user to specify where a flux is
    located.

    If a reaction is dependent on species that are not in the LHS or RHS they
    are placed into speciesDict 

    track_value allows tracking/recording of the flux values for a particular
    species involved in a reaction. e.g. in a reaction A+X <-> B we may be
    interested in how much of B is created, in that reaction "B" would be the
    track_value
    """
    def __init__(self):
        self.df = pd.DataFrame(columns=["group", "LHS", "RHS",
                                        "paramDict", "reaction_type",
                                        "explicit_restriction_to_domain", "speciesDict",
                                        "track_value"])
        self.df = pd.DataFrame({'group': pd.Series([], dtype=str),
                                'LHS': pd.Series([], dtype=object),
                                'RHS': pd.Series([], dtype=object),
                                'paramDict': pd.Series([], dtype=object),
                                'reaction_type': pd.Series([], dtype=str),
                                'explicit_restriction_to_domain': pd.Series([], dtype=object),
                                'speciesDict': pd.Series([], dtype=object),
                                'track_value': pd.Series([], dtype=bool)})
    def append(self, name, group, LHS, RHS, paramDict,
               reaction_type="mass_action", explicit_restriction_to_domain=None,
               speciesDict={}, track_value=False):

        self.df = self.df.append(pd.Series({"group": group, "LHS": LHS,
                                  "RHS": RHS, "paramDict": paramDict,
                                  "reaction_type": reaction_type,
                                  "explicit_restriction_to_domain": explicit_restriction_to_domain,
                                  "speciesDict": speciesDict,
                                  "track_value": track_value}, name=name))#, ignore_index=True)

    def write_json(self, name='reactions.json'):
        self.df.to_json(name)
        print("Reactions generated successfully! Saved as %s" % name)




