# """
# Classes/functions used to construct models
# """
# from stubs import unit

# import pandas
# import dolfin as d
# import pint 

# # IN PROGRESS
# # class ObjectDF:
# #     def __init__(self, property_types):
# #         """General dataframe wrapper class which can hold parameters, species, etc.
        
# #         Args:
# #             property_types (dict): property_name -> data type. Each key/value
# #             pair will correspond to a pandas Series.
# #         """
# #         self.property_types
# #         data_dict = dict()
# #         for property_name, property_type in property_types.items():
# #             data_dict.update({property_name, 
# #                              pandas.Series([], dtype=property_type)})
# #         self.df = pandas.DataFrame(data_dict)

# #         def add(self, name, **kwargs):
# #             inputs = locals()
# #             to_df = kwargs.update({"name": name})
# #             # ensure types are correct
# #             for key, value in to_df.items():
# #                 to_df[key] =  

# class ObjectDF:
#     """General dataframe wrapper class which can hold parameters, species, etc."""
#     def __init__(self):
#         pass


# class ParameterDF(ObjectDF):
#     """
#     A standard (non time-dependent) parameter has an associated value, unit,
#     group [i.e. what physics does the parameter belong to?], notes, and a 
#     dolfin_constant [a dolfin Constant class with value being adjusted for unit
#     consistency and/or time-dependency].
#     A time-dependent parameter can either be expressed as a symbolic expression
#     wrt t or it can be sampled from some dataset (a .csv file with time as the
#     first column and values as the second column). If the parameter is provided
#     as a symbolic expression it can also be provided in pre-integrated format,
#     an expression which is defined as 

#     \f$p_I(t) = \int_0^t p(x) \,dx \f$

#     This is useful for enabling larger time-steps when using functions with
#     rapidly changing values.
#     """
#     def __init__(self):
#         self.df = pandas.DataFrame({'value': pandas.Series([], dtype=float),
#                                 'unit': pandas.Series([], dtype=object),
#                                 'group': pandas.Series([], dtype=str),
#                                 'notes': pandas.Series([], dtype=str),
#                                 'is_time_dependent': pandas.Series([], dtype=bool),
#                                 # is the data sampled from some discrete data?
#                                 'sampling_file': pandas.Series([], dtype=str), 
#                                 'sampling_data': pandas.Series([], dtype=object), 
#                                 'preint_sampling_data': pandas.Series([], dtype=object), 
#                                 'dolfin_constant': pandas.Series([], dtype=object),
#                                 'sym_expr': pandas.Series([], dtype=object),
#                                 'preint_sym_expr': pandas.Series([], dtype=object)})

#     def append(self, name, value, unit, group, notes='', is_time_dependent=False,
#                sampling_file='', sampling_data=None, dolfin_constant=None,
#                sym_expr=None, preint_sampling_data=None, preint_sym_expr=None):
#         """
#         Adds data to the parameter
#         """

#         # check that the parameter is not under or over specified
#         if is_time_dependent:
#             if sym_expr and sampling_file:
#                 raise Exception("Provide parameter %s with either a symbolic"\
#                                 "expression or discrete data, not both!" % name)
#             if not (sym_expr or sampling_file):
#                 raise Exception("Time-dependent parameter %s must either have"\
#                                 "a sympy expression or have discrete data!"\
#                                 % name)

#         self.df = self.df.append(pandas.Series({"value": value, "unit": str(unit),
#                                             "group": group, "notes": notes,
#                                             "is_time_dependent": is_time_dependent,
#                                             # is the data sampled from some discrete data?
#                                             'sampling_file': sampling_file,
#                                             'sampling_data': sampling_data,
#                                             'preint_sampling_data': preint_sampling_data,
#                                             "dolfin_constant": dolfin_constant,
#                                             "sym_expr": sym_expr,
#                                             "preint_sym_expr": preint_sym_expr},
#                                             name=name))

#     def write_json(self, name='parameters.json'):
#         self.df.to_json(name)
#         print("Parameters generated successfully! Saved as %s" % name)



# class SpeciesDF(ObjectDF):
#     """
#     IC assumed to be in terms concentration units
#     """
#     def __init__(self):
#         self.df = pandas.DataFrame({'initial_condition': pandas.Series([], dtype=float),
#                                 'concentration_units': pandas.Series([], dtype=object),
#                                 'D': pandas.Series([], dtype=float),
#                                 'diffusion_units': pandas.Series([], dtype=object),
#                                 'compartment_name': pandas.Series([], dtype=str),
#                                 'group': pandas.Series([], dtype=str)})

#     def append(self, name, group, units, IC, D, diffusion_units, compartment_name):
#         self.df = self.df.append(pandas.Series({"group": group,
#                                             "concentration_units": str(units),
#                                             "initial_condition": IC, 
#                                             "D": D,
#                                             "diffusion_units": str(diffusion_units),
#                                             "compartment_name": compartment_name}, name=name))

#     def write_json(self, name='species.json'):
#         self.df.to_json(name)
#         print("Species generated successfully! Saved as %s" % name)

# class CompartmentDF(ObjectDF):
#     """
#     Dimensionality refers to the topological dimension (e.g. a triangle is always 2d regardless it is embedded
#     in R^2 or R^3.
#     """
#     def __init__(self):
#         self.df = pandas.DataFrame({'dimensionality': pandas.Series([], dtype=int),
#                                 'compartment_units': pandas.Series([], dtype=object),
#                                 'cell_marker': pandas.Series([], dtype=object)})

#     def append(self, name, dim, units, marker):
#         self.df = self.df.append(pandas.Series({"dimensionality": dim,
#                                            "compartment_units": str(units),
#                                             "cell_marker": marker}, name=name))

#     def write_json(self, name='compartments.json'):
#         self.df.to_json(name)
#         print("Compartment generated successfully! Saved as %s" % name)


# class ReactionDF(ObjectDF):
#     """
#     A reaction is specified by the reactants/products on the left hand side 
#     (lhs), right hand side (rhs), kinetic parameters, the reaction type (STUBS
#     will always assume mass action unless specified otherwise).

#     If a reaction contains species from two different compartments
#     then STUBS will automatically apply boundary conditions and scale the fluxes
#     appropriately. However sometimes a flux is dependent only on a species in
#     one compartment yet is still a boundary flux -
#     explicit_restriction_to_domain allows the user to specify where a flux is
#     located.

#     If a reaction is dependent on species that are not in the lhs or rhs they
#     are placed into species_map 

#     track_value allows tracking/recording of the flux values for a particular
#     species involved in a reaction. e.g. in a reaction A+X <-> B we may be
#     interested in how much of B is created, in that reaction "B" would be the
#     track_value
#     """
#     def __init__(self):
#         self.df = pandas.DataFrame(columns=["group", "lhs", "rhs",
#                                         "param_map", "reaction_type",
#                                         "explicit_restriction_to_domain", "species_map",
#                                         "track_value"])
#         self.df = pandas.DataFrame({'group': pandas.Series([], dtype=str),
#                                 'lhs': pandas.Series([], dtype=object),
#                                 'rhs': pandas.Series([], dtype=object),
#                                 'param_map': pandas.Series([], dtype=object),
#                                 'reaction_type': pandas.Series([], dtype=str),
#                                 'explicit_restriction_to_domain': pandas.Series([], dtype=object),
#                                 'species_map': pandas.Series([], dtype=object),
#                                 'track_value': pandas.Series([], dtype=bool)})
#     def append(self, name, group, lhs, rhs, param_map,
#                reaction_type="mass_action", explicit_restriction_to_domain=None,
#                species_map={}, track_value=False):

#         self.df = self.df.append(pandas.Series({"group": group, "lhs": lhs,
#                                   "rhs": rhs, "param_map": param_map,
#                                   "reaction_type": reaction_type,
#                                   "explicit_restriction_to_domain": explicit_restriction_to_domain,
#                                   "species_map": species_map,
#                                   "track_value": track_value}, name=name))#, ignore_index=True)

#     def write_json(self, name='reactions.json'):
#         self.df.to_json(name)
#         print("Reactions generated successfully! Saved as %s" % name)

# def empty_sbmodel(output_type=tuple):
#     """
#     Convenience function. Gives an empty sbmodel
#     """
#     if output_type==tuple:
#         return (ParameterDF(), SpeciesDF(),
#                 CompartmentDF(), ReactionDF())
#     elif output_type==dict:
#         return {'parameter_container': ParameterDF(),
#                 'species_container': SpeciesDF(),
#                 'compartment_container': CompartmentDF(),
#                 'reaction_container': ReactionDF()}
#     else:
#         raise ValueError("Unknown output type.")


