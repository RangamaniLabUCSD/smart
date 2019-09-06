import dolfin as d
import pandas as pd
import pint
import sympy
from sympy.parsing.sympy_parser import parse_expr
from termcolor import colored

from stubs import unit
#import stubs.model_assembly as model_assembly



#example call
# customEqn(sdf,pdf,symStr)

# custom reaction laws

"""
most fundamental information needed for a flux:
- flux name
- species name (the species that this flux is being applied to)
- parameters and species (pdf,sdf), dfDict
- stoichiometry (scalar. includes sign)
- form of equation (symStr -> sympy lambda expression)
- location (often can be inferred from species in reaction. other times must be specified explicitly)

information that can be obtained aposteriori
- involved compartments
- flux dimensionality
- boundary marker
- flux units
- group
- dx, ds, dP
- v

"""


#     def getAdditionalFluxProperties(self):
#         # get additional properties of the flux
#         self.involved_compartments = self.getInvolvedCompartments()
#         self.j['involved_compartments'] = list(self.involved_compartments.keys())
#         self.j['source_compartment'], self.j['destination_compartment'] = self.getSourceDestination()
#         self.j['flux_dimensionality'] = self.getFluxDimensionality()
#         self.j['boundary_marker'] = self.getBoundaryMarker()
#         self.j['flux_units'] = self.getFluxUnits()
#         self.j['is_linear_wrt'] = self.getIsLinear()
#         self.j['is_linear_wrt_comp'] = self.getIsLinearComp()



#     def getSourceDestination(self):
#         # which compartment is the source, and which is the destination
#         compList = list(self.j['involved_compartments'])
#         destinationSp = self.j['species_name']
#         destination_compartment = self.involved_species[destinationSp]['compartment_name']
#         compList.remove(destination_compartment)
#         if len(compList)==0:
#             source_compartment = destination_compartment
#         else:
#             source_compartment = compList.pop()

#         return [source_compartment, destination_compartment]

#     def getFluxDimensionality(self):
#         # dimensionality from source -> destination
#         sourceDim = self.involved_compartments[self.j['source_compartment']].squeeze().dimensionality
#         destinationDim = self.involved_compartments[self.j['destination_compartment']].squeeze().dimensionality
# #        sourceDim = cdf.loc[cdf.compartment_name==j.['source_compartment']].squeeze().dimensionality
# #        destinationDim = cdf.loc[cdf.compartment_name==j.['destination_compartment']].squeeze().dimensionality
#         return [sourceDim, destinationDim]

#     def getBoundaryMarker(self):
#         dim = self.j['flux_dimensionality']
#         if dim[1] <= dim[0]:
#             boundary_marker = None
#         elif dim[1] > dim[0]: # boundary flux
#             boundary_marker = self.involved_compartments[self.j['source_compartment']].squeeze().cell_marker
#         return boundary_marker

#     def getFluxUnits(self):
#         destinationSp = self.involved_species[self.j['species_name']].squeeze()
#         if self.j['boundary_marker']:
#             flux_units = destinationSp.concentration_units / destinationSp.compartment_units * destinationSp.D_units
#         else:
#             flux_units = destinationSp.concentration_units / unit.s
#         return flux_units

#     def getIsLinear(self):
#         """
#         For a given flux we want to know which terms are linear
#         """
#         is_linear_wrt = {}
#         for symVar in self.j['symEqn'].free_symbols:
#             varName = str(symVar)
#             if varName in self.j['involved_species']:
#                 if sympy.diff(self.j['symEqn'], varName , 2).is_zero:
#                     is_linear_wrt[varName] = True
#                 else:
#                     is_linear_wrt[varName] = False
#         return is_linear_wrt

#     def getIsLinearComp(self):
#         """
#         Is the flux linear in terms of a component vector (e.g. dj/du['pm'])
#         """
#         is_linear_wrt_comp = {}
#         ucomp = {}
#         umap = {}
#         for compName in self.j['involved_compartments']:
#             ucomp[compName] = 'u'+compName

#         for symVar in self.j['symEqn'].free_symbols:
#             varName = str(symVar)
#             if varName in self.j['involved_species']:
#                 compName = self.sdf.loc[self.sdf.species_name==varName,'compartment_name'].squeeze()
#                 umap.update({varName: ucomp[compName]})


#         for compName in self.j['involved_compartments']:
#             newEqn = self.j['symEqn'].subs(umap)
#             if sympy.diff(newEqn, ucomp[compName], 2).is_zero:
#                 is_linear_wrt_comp[compName] = True
#             else:
#                 is_linear_wrt_comp[compName] = False

#         return is_linear_wrt_comp





































# #==============================================================================





#     class customReaction(object):
#         def __init__(self, symStr):

#             symEqn = parse_expr(symStr)
#             self.varList = symEqn.free_symbols
#             self.symStr = {'base': symStr}
#             self.symEqn = {'base': symEqn}

#         def createFlux(self, flux_name, species_name, stoichiometry, varDict, sdf, cdf, pdf,
#                        group='custom', flux_type='custom', explicit_restriction_to_domain=None):
#             # map varDict variables into symStr
#             if flux_name in self.symEqn.keys():
#                 raise Exception("Flux name must be unique!")
#             self.symEqn[flux_name] = self.symEqn['base'].subs(varDict)
#             self.symStr[flux_name] = str(self.symEqn[flux_name])

#             flux = Flux(flux_name, species_name, self.symStr[flux_name], stoichiometry, group, flux_type,
#                         sdf, cdf, pdf, explicit_restriction_to_domain)
#             flux.getAdditionalFluxProperties()
#             return flux

#         def getFluxes(self, rxn, sdf, cdf, pdf):
#             reactantList = [sdf[sdf.species_name==sp].squeeze() for sp in rxn.reactants]
#             productList = [sdf[sdf.species_name==sp].squeeze() for sp in rxn.products]
#             speciesList = reactantList+productList
#             stoichList = rxn.stoich_r + rxn.stoich_p
#             fluxList = []

#             for idx, sp in enumerate(speciesList):
#                 stoichiometry = stoichList[idx]
#                 isReactant = sp.species_name in rxn.reactants
#                 sign = -1 if isReactant else 1
#                 signedStoichiometry = sign*stoichiometry

#                 flux_name = rxn.reaction_name + ' [' + sp.species_name + ']'
#                 #jserca = hillI.createFlux("Ca -> 0 [serca]", 'Ca', -1, {'u':'Ca', 'k': 'kserca', 'n': 'nserca', 'km': 'kmserca'}, sdf, cdf, pdf, explicit_restriction_to_domain='er')
#                 rxn.paramDict.update(rxn.speciesDict)

#                 fluxList.append(self.createFlux(flux_name, sp.species_name, signedStoichiometry, rxn.paramDict, sdf, cdf, pdf,
#                     group=rxn.group, explicit_restriction_to_domain=rxn.explicit_restriction_to_domain))

#             return fluxList
# class Flux(object):
#     def __init__(self, flux_name, species_name, symStr, stoichiometry, group, flux_type, sdf, cdf, pdf, explicit_restriction_to_domain=None):
# #        j = pd.DataFrame(columns=["flux_name", "species_name", "involved_species", "involved_parameters", "involved_compartments", "stoichiometry",
# #                            "symEqn", "lambdaEqn", "flux", "flux_dimensionality", "boundary_marker", "flux_units", "group", "flux_type"])
# # group, flux_type
#         j = {}
#         symEqn = parse_expr(symStr)
#         symList = symEqn.free_symbols
#         varList = [str(x) for x in symList] + [species_name]
#         involved_species = {}
#         involved_parameters = {}
#         for var in varList:
#             s = sdf.species_name==var
#             p = pdf.parameter_name==var
#             if len(s[s].index) + len(p[p].index) != 1:
#                 raise Exception("Variable %s has %d species/parameter(s) with the same name (must be unique)."%(var,len(s[s].index) + len(p[p].index)))
#             if len(s[s].index)==1:
#                 involved_species[var] = sdf.iloc[s[s].index[0]]
#             elif len(p[p].index)==1:
#                 involved_parameters[var] = pdf.iloc[p[p].index[0]]
#         j['flux_name'] = flux_name
#         j['species_name'] = species_name
#         j['symEqn'] = symEqn
#         j['stoichiometry'] = stoichiometry
#         j['group'] = group
#         j['flux_type'] = flux_type
#         j['lambdaEqn'] = sympy.lambdify(symList, symEqn)
#         j['involved_species'] = list(involved_species.keys())
#         j['involved_parameters'] = list(involved_parameters.keys())
#         self.involved_species = involved_species
#         self.involved_parameters = involved_parameters
#         j['explicit_restriction_to_domain'] = explicit_restriction_to_domain
#         self.j = j
#         self.sdf = sdf
#         self.cdf = cdf
#         self.pdf = pdf
#     def getAdditionalFluxProperties(self):
#         # get additional properties of the flux
#         self.involved_compartments = self.getInvolvedCompartments()
#         self.j['involved_compartments'] = list(self.involved_compartments.keys())
#         self.j['source_compartment'], self.j['destination_compartment'] = self.getSourceDestination()
#         self.j['flux_dimensionality'] = self.getFluxDimensionality()
#         self.j['boundary_marker'] = self.getBoundaryMarker()
#         self.j['flux_units'] = self.getFluxUnits()
#         self.j['is_linear_wrt'] = self.getIsLinear()
#         self.j['is_linear_wrt_comp'] = self.getIsLinearComp()
#     def getInvolvedCompartments(self):
#         speciesList = list(self.involved_species.values())
#         print(self.j['flux_name'])
#         #print(speciesList)
#         cdf = self.cdf
#         # find involved compartments
#         involved_compartments = {}
#         for sp in self.involved_species.values():
#             compName = sp.squeeze().compartment_name
#             if compName not in involved_compartments.keys():
#                 involved_compartments[compName] = cdf.loc[cdf.compartment_name==compName].squeeze()
#         if isinstance(self.j['explicit_restriction_to_domain'], str):
#             compName = self.j['explicit_restriction_to_domain']
#             if compName not in involved_compartments.keys():
#                 involved_compartments[compName] = cdf.loc[cdf.compartment_name==compName].squeeze()
#         if len(involved_compartments.keys()) not in (1,2):
#             raise Exception("Number of compartments involved in a flux must be either one or two!")
#         return involved_compartments
#     def getSourceDestination(self):
#         # which compartment is the source, and which is the destination
#         compList = list(self.j['involved_compartments'])
#         destinationSp = self.j['species_name']
#         destination_compartment = self.involved_species[destinationSp]['compartment_name']
#         compList.remove(destination_compartment)
#         if len(compList)==0:
#             source_compartment = destination_compartment
#         else:
#             source_compartment = compList.pop()
#         return [source_compartment, destination_compartment]
#     def getFluxDimensionality(self):
#         # dimensionality from source -> destination
#         sourceDim = self.involved_compartments[self.j['source_compartment']].squeeze().dimensionality
#         destinationDim = self.involved_compartments[self.j['destination_compartment']].squeeze().dimensionality
# #        sourceDim = cdf.loc[cdf.compartment_name==j.['source_compartment']].squeeze().dimensionality
# #        destinationDim = cdf.loc[cdf.compartment_name==j.['destination_compartment']].squeeze().dimensionality
#         return [sourceDim, destinationDim]
#     def getBoundaryMarker(self):
#         dim = self.j['flux_dimensionality']
#         if dim[1] <= dim[0]:
#             boundary_marker = None
#         elif dim[1] > dim[0]: # boundary flux
#             boundary_marker = self.involved_compartments[self.j['source_compartment']].squeeze().cell_marker
#         return boundary_marker
#     def getFluxUnits(self):
#         destinationSp = self.involved_species[self.j['species_name']].squeeze()
#         if self.j['boundary_marker']:
#             flux_units = destinationSp.concentration_units / destinationSp.compartment_units * destinationSp.D_units
#         else:
#             flux_units = destinationSp.concentration_units / unit.s
#         return flux_units
#     def getIsLinear(self):
        
#         For a given flux we want to know which terms are linear
        
#         is_linear_wrt = {}
#         for symVar in self.j['symEqn'].free_symbols:
#             varName = str(symVar)
#             if varName in self.j['involved_species']:
#                 if sympy.diff(self.j['symEqn'], varName , 2).is_zero:
#                     is_linear_wrt[varName] = True
#                 else:
#                     is_linear_wrt[varName] = False
#         return is_linear_wrt
#     def getIsLinearComp(self):
#         """
#         Is the flux linear in terms of a component vector (e.g. dj/du['pm'])
#         """
#         is_linear_wrt_comp = {}
#         ucomp = {}
#         umap = {}
#         for compName in self.j['involved_compartments']:
#             ucomp[compName] = 'u'+compName
#         for symVar in self.j['symEqn'].free_symbols:
#             varName = str(symVar)
#             if varName in self.j['involved_species']:
#                 compName = self.sdf.loc[self.sdf.species_name==varName,'compartment_name'].squeeze()
#                 umap.update({varName: ucomp[compName]})
#         for compName in self.j['involved_compartments']:
#             newEqn = self.j['symEqn'].subs(umap)
#             if sympy.diff(newEqn, ucomp[compName], 2).is_zero:
#                 is_linear_wrt_comp[compName] = True
#             else:
#                 is_linear_wrt_comp[compName] = False
#         return is_linear_wrt_comp
# class customReaction(object):
#     def __init__(self, symStr):
#         symEqn = parse_expr(symStr)
#         self.varList = symEqn.free_symbols
#         self.symStr = {'base': symStr}
#         self.symEqn = {'base': symEqn}
#     def createFlux(self, flux_name, species_name, stoichiometry, varDict, sdf, cdf, pdf,
#                    group='custom', flux_type='custom', explicit_restriction_to_domain=None):
#         # map varDict variables into symStr
#         if flux_name in self.symEqn.keys():
#             raise Exception("Flux name must be unique!")
#         self.symEqn[flux_name] = self.symEqn['base'].subs(varDict)
#         self.symStr[flux_name] = str(self.symEqn[flux_name])
#         flux = Flux(flux_name, species_name, self.symStr[flux_name], stoichiometry, group, flux_type,
#                     sdf, cdf, pdf, explicit_restriction_to_domain)
#         flux.getAdditionalFluxProperties()
#         return flux
#     def getFluxes(self, rxn, sdf, cdf, pdf):
#         reactantList = [sdf[sdf.species_name==sp].squeeze() for sp in rxn.reactants]
#         productList = [sdf[sdf.species_name==sp].squeeze() for sp in rxn.products]
#         speciesList = reactantList+productList
#         stoichList = rxn.stoich_r + rxn.stoich_p
#         fluxList = []
#         for idx, sp in enumerate(speciesList):
#             stoichiometry = stoichList[idx]
#             isReactant = sp.species_name in rxn.reactants
#             sign = -1 if isReactant else 1
#             signedStoichiometry = sign*stoichiometry
#             flux_name = rxn.reaction_name + ' [' + sp.species_name + ']'
#             #jserca = hillI.createFlux("Ca -> 0 [serca]", 'Ca', -1, {'u':'Ca', 'k': 'kserca', 'n': 'nserca', 'km': 'kmserca'}, sdf, cdf, pdf, explicit_restriction_to_domain='er')
#             rxn.paramDict.update(rxn.speciesDict)
#             fluxList.append(self.createFlux(flux_name, sp.species_name, signedStoichiometry, rxn.paramDict, sdf, cdf, pdf,
#                 group=rxn.group, explicit_restriction_to_domain=rxn.explicit_restriction_to_domain))
#         return fluxList
# def reactionToFluxes(rxn, sdf, cdf, pdf):
#     reactantList = [sdf[sdf.species_name==sp].squeeze() for sp in rxn.reactants]
#     productList = [sdf[sdf.species_name==sp].squeeze() for sp in rxn.products]
#     speciesList = reactantList+productList
#     stoichList = rxn.stoich_r + rxn.stoich_p
#     fluxList = []
#     for idx, sp in enumerate(speciesList):
#         stoichiometry = stoichList[idx]
#         if rxn.reaction_type == 'mass_action':
#             if reactantList:
#                 fluxList.append(flux_massActionForward(sp, reactantList, stoichiometry, rxn, sdf, cdf, pdf))
#             if productList:
#                 fluxList.append(flux_massActionReverse(sp, productList, stoichiometry, rxn, sdf, cdf, pdf))
#         elif rxn.reaction_type == 'mass_action_forward':
#             if reactantList:
#                 fluxList.append(flux_massActionForward(sp, reactantList, stoichiometry, rxn, sdf, cdf, pdf))
# #        elif rxn.reaction_type == 'hilli':
# #            fluxList.append(flux_hillIrreversible(sp, speciesList, stoichiometry, rxn, sdf, cdf, pdf))
#         elif rxn.reaction_type == 'hillI':
#             fluxList.extend(hillI.getFluxes(rxn, sdf, cdf, pdf))
#         elif rxn.reaction_type == 'ip3r':
#             fluxList.extend(ip3r.getFluxes(rxn, sdf, cdf, pdf))
#         elif rxn.reaction_type == 'dhdt':
#             fluxList.extend(dhdt.getFluxes(rxn, sdf, cdf, pdf))
#     for flux in fluxList:
#         flux.getAdditionalFluxProperties()
#     return fluxList
# def flux_massActionForward(sp, reactantList, stoichiometry, rxn, sdf, cdf, pdf):
#     isReactant = sp.species_name in rxn.reactants
#     sign = -1 if isReactant else 1
#     signedStoichiometry = sign*stoichiometry
#     symStr = rxn.paramDict['on']
#     for ridx, rsp in enumerate(reactantList):
#         symStr += '*' + rsp.species_name
#         if rxn.stoich_r[ridx] != 1:
#             symStr += '**' + rxn.stoich_r[ridx]
#     flux_name = rxn.reaction_name + " [" + sp.species_name + ": forward]"
#     return Flux(flux_name, sp.species_name, symStr, signedStoichiometry, rxn.group, 'mass_action_forward',
#                 sdf, cdf, pdf, rxn.explicit_restriction_to_domain)
# def flux_massActionReverse(sp, productList, stoichiometry, rxn, sdf, cdf, pdf):
#     isReactant = sp.species_name in rxn.reactants
#     sign = 1 if isReactant else -1
#     signedStoichiometry = sign*stoichiometry
#     symStr = rxn.paramDict['off']
#     for pidx, psp in enumerate(productList):
#         symStr += '*' + psp.species_name
#         if rxn.stoich_p[pidx] != 1:
#             symStr += '**' + rxn.stoich_p[pidx]
#     flux_name = rxn.reaction_name + " [" + sp.species_name + ": reverse]"
#     return Flux(flux_name, sp.species_name, symStr, signedStoichiometry, rxn.group, 'mass_action_reverse',
#                 sdf, cdf, pdf, rxn.explicit_restriction_to_domain)
# # "custom reactions"
# hillI = customReaction('k * u**n / (u**n + km)')
# ip3r = customReaction('kip3r * (1-Ca/CaER) * (h*Ca*IP3/((Ca+d_Ca)*(IP3+d_IP3)))**3')
# dhdt = customReaction('(K1-(Ca+K1)*h)*K2')
# def flux_to_dolfin(flux, sdf, pdf):
#     # find the relevant parameters/species to this equation
#     valueDict = {}
#     unitDict = {}
#     flux_species = sdf.loc[sdf.species_name==flux.species_name].squeeze()
#     for symVar in flux.symEqn.free_symbols:
#         varName = str(symVar)
#         param = pdf.loc[pdf.parameter_name==varName].squeeze()
#         species = sdf.loc[sdf.species_name==varName].squeeze()
#         if not param.empty:
#             if param.is_time_dependent:
#                 valueDict[varName] = param.dolfinConstant.get()
#             else:
#                 valueDict[varName] = param.value_unit.magnitude
#             unitDict[varName] = param.value_unit
#         elif not species.empty:
#             #ukey = get_ukey(flux.flux_dimensionality, species, flux.is_linear_wrt[varName])
#             ukey = get_ukey(flux, flux_species, species)
#             print('flux %s: ukey %s used for variable %s'%(flux.flux_name, ukey, varName))
#             valueDict[varName] = species.u[ukey].get()
#             unitDict[varName] = species.concentration_units
#     #return (paramDict, valueDict,unitDict)
#     #print()
#     # assemble the terms using the symbolic algebra
#     prod = flux.lambdaEqn(**valueDict)
#     unitProd = flux.lambdaEqn(**unitDict)
#     unitProd = 1 * (1*unitProd).units # trick to make object a "Quantity" class
#     return (prod,unitProd)
# def assemble_fluxes(sdf, pdf, cdf, jdf):
#     """
#     Creates the actual dolfin objects for each flux. Checks units for consistency
#     """
#     for jidx, j in jdf.iterrows():
#         prod, unitProd = flux_to_dolfin(j, sdf, pdf)
#         # first, check unit consistency
#         if (unitProd.units/j.flux_units).dimensionless:
#             pass
#         else:
#             print(j.flux_name)
#             print(j.involved_compartments)
#             print('Adjusting flux between compartments %s and %s by length scale factor to ensure consistency' % tuple(j.involved_compartments))
#             comp1 = cdf.loc[cdf.compartment_name==j.involved_compartments[0]].squeeze()
#             lengthScale = comp1.scale_to[j.involved_compartments[1]]
#             if (lengthScale*unitProd/j.flux_units).dimensionless:
#                 print("Yes")
#                 prod *= lengthScale.magnitude
#                 unitProd = unitProd * lengthScale.units
#             elif (1/lengthScale*unitProd/j.flux_units).dimensionless:
#                 print("No")
#                 prod /= lengthScale.magnitude
#                 unitProd /= lengthScale.units
#             else:
#                 print('Hey!')
#                 print(j.flux_name)
#                 print(lengthScale)
#                 print(unitProd)
#                 print(j.flux_units)
#                 print((lengthScale*unitProd/j.flux_units))
#                 print((1/lengthScale*unitProd/j.flux_units))
#                 raise Exception("Inconsitent units!")
#             print('unitProd post: %s' % unitProd.units)
#         # if units are consistent in dimensionality but not magnitude, adjust values
#         if j.flux_units != unitProd:
#             printStr = ('\nThe flux, %s, has units '%j.flux_name + colored(unitProd, "red") +
#                 "...the desired units for this flux are " + colored(j.flux_units, "cyan"))
#             print(printStr)
#             unitScaling = unitProd.to(j.flux_units).magnitude
#             prod *= unitScaling
#             print('Adjusted value of flux by ' + colored("%f"%unitScaling, "cyan") + ' to match units.\n')
#         # adjust sign/stoichiometry if necessary
#         prod *= j.stoichiometry
#         jdf.at[jidx, 'flux'] = model_assembly.ref(prod)

# def get_ukey(flux, flux_species, varSpecies):
#     """
#     Given the dimensionality of a flux (e.g. 2d surface to 3d vol) and the dimensionality
#     of a species, determine which term of u should be used
#     flux_species: the species this flux is being applied to
#     varSpecies: the variable in the flux that we are checking the key for
#     """
#     flux_species_dimensionality = flux_species.dimensionality
#     flux_species_comp = flux_species.compartment_name
#     flux_species_name = flux_species.species_name
#     var_dimensionality = varSpecies.dimensionality
#     var_comp = varSpecies.compartment_name
#     var_name = varSpecies.species_name
#     if flux_species_name == var_name:
#         if flux.is_linear_wrt[flux_species_name]:
#             return 't'
#         else:
#             return 'k'
#     if flux_species_comp == var_comp:
#         if flux.is_linear_wrt_comp[var_comp]:
#             return 't'
#         else:
#             return 'k'
#     if var_dimensionality < flux_species_dimensionality:
#         return 'u'
#     if var_dimensionality > flux_species_dimensionality:
#         return 'b'
#     raise Exception("If you made it to this far in get_ukey() I missed some logic...")
# # should be able to
# # return prod, unitProd, ukey, isLinear
