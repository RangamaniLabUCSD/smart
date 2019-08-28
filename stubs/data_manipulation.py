# Functions to help with managing solutions / post-processing
import dolfin as d
import numpy as np
import matplotlib.pyplot as plt
import pickle


from stubs import unit
import stubs.model_assembly as model_assembly

# matplotlib settings
lwsmall = 1.5
lwmed = 3
lineopacity = 0.6
fsmed = 7
fssmall = 5

class data(object):
    def __init__(self, model_parameters):
        self.model_parameters = model_parameters
        self.solutions = {}
        self.errors = {}
        #self.plots = {'solutions': {'fig': plt.figure(), 'subplots': []}}
        self.plots = {'solutions': plt.figure()}
        self.timer = d.Timer()

    def initVTK(self, sdf, speciesList):
        for spName in speciesList:
            self.solutions[spName] = {}
            sp = sdf.loc[sdf.species_name==spName].squeeze()
            #compName = sp.compartment_name
            #compIdx = int(sp.compartment_index)
            #num_species = sp.num_species
            #concentration_units = sp.concentration_units

            self.solutions[spName]['num_species'] = sp.num_species
            self.solutions[spName]['compName'] = sp.compartment_name
            self.solutions[spName]['compIdx'] = int(sp.compartment_index)
            self.solutions[spName]['concentration_units'] = sp.concentration_units

            fileStr = self.model_parameters['solnDir'] + '/' + spName + '.pvd'
            self.solutions[spName]['vtk'] = d.File(fileStr)

    def storeVTK(self, u, t):
        for spName in self.solutions.keys():
            compName = self.solutions[spName]['compName']
            compIdx = self.solutions[spName]['compIdx']
            if self.solutions[spName]['num_species'] == 1:
                self.solutions[spName]['vtk'] << (u[compName]['u'], t)
            else:
                self.solutions[spName]['vtk'] << (u[compName]['u'].split()[compIdx], t)

    def computeStatistics(self, u, t, V, speciesList):

        for spName in speciesList:
            compName = self.solutions[spName]['compName']
            compIdx = self.solutions[spName]['compIdx']

            # compute statistics and append values
            ustats = dolfinGetFunctionStats(u[compName]['u'], V[compName], compIdx)
            for key, value in ustats.items():
                if key not in self.solutions[spName].keys():
                    self.solutions[spName][key] = [value]
                else:
                    self.solutions[spName][key].append(value)

            # append the time
            if 'tvec' not in self.solutions[spName].keys():
                self.solutions[spName]['tvec'] = [t]
            else:
                self.solutions[spName]['tvec'].append(t)

    def computeError(self, u, compName, errorNormKeys):
        errorNormDict = {'L2': 2, 'Linf': np.Inf}
        if compName not in self.errors.keys():
            self.errors[compName] = {}
        for key in errorNormKeys:
            if key not in self.errors[compName].keys():
                self.errors[compName][key] = [np.linalg.norm(u[compName]['u'].vector().get_local()
                                        - u[compName]['k'].vector().get_local(), ord=errorNormDict[key])]
            else:
                self.errors[compName][key].append(np.linalg.norm(u[compName]['u'].vector().get_local()
                                        - u[compName]['k'].vector().get_local(), ord=errorNormDict[key]))

    def initPlot(self):
        # NOTE: for now plots are individual; add an option to group species into subplots by category
        numPlots = len(self.solutions.keys())
        subplotCols = 3
        subplotRows = int(np.ceil(numPlots/subplotCols))
        for idx, key in enumerate(self.solutions.keys()):
            self.plots['solutions'].add_subplot(subplotRows,subplotCols,idx+1)


    def plotSolutions(self):
        for idx, key in enumerate(self.solutions.keys()):
            soln =  self.solutions[key]
            subplot = self.plots['solutions'].get_axes()[idx]
            subplot.clear()
            subplot.plot(soln['tvec'], soln['min'], linewidth=lwsmall, color='k')
            subplot.plot(soln['tvec'], soln['mean'], linewidth=lwmed, color='k')
            subplot.plot(soln['tvec'], soln['max'], linewidth=lwsmall, color='k')

            unitStr = '{:P}'.format(self.solutions[key]['concentration_units'].units)
            subplot = self.plots['solutions'].get_axes()[idx]
            subplot.title.set_text(key)# + ' [' + unitStr + ']')
            subplot.title.set_fontsize(fsmed)
            #self.plots['solutions'].canvas.draw()
            #self.plots['solutions'].show()

        #self.plots['solutions'].tight_layout()
        for ax in self.plots['solutions'].axes:
            ax.ticklabel_format(useOffset=False)
            plt.setp(ax.get_xticklabels(), fontsize=fssmall)
            plt.setp(ax.get_yticklabels(), fontsize=fssmall)
        self.plots['solutions'].savefig(self.model_parameters['figName'],dpi=300,bbox_inches='tight')

    def outputPickle(self):
        saveKeys = ['tvec','min','mean','max','std']
        newDict = {}
        for spName in self.solutions.keys():
            newDict[spName] = {}
            for key in saveKeys:
                newDict[spName][key] = self.solutions[spName][key]

        # pickle file
        with open('solutions_'+self.model_parameters['tag']+'.obj', 'wb') as pickle_file:
            pickle.dump(newDict, pickle_file)

#    def finalizePlot(self):
#        for idx, key in enumerate(self.solutions.keys()):
#            unitStr = '{:P}'.format(self.solutions[key]['concentration_units'].units)
#            subplot = self.plots['solutions'].get_axes()[idx]
#            subplot.title.set_text(key + ' [' + unitStr + ']')
#            subplot.title.set_fontsize(fsmed)
#        self.plots['solutions'].savefig(self.model_parameters['figName'])





#import matplotlib.pyplot as plt
#import time
#fig = plt.figure()
#fig.show()
#for i in range(10):
#    plt.clf()
#    xlist = list(range(i))
#    ylist = [x**2 for x in xlist]
#    plt.plot(xlist,ylist)
#    time.sleep(0.3)
#
#    fig.canvas.draw()
#
#import matplotlib.pyplot as pylab
#dat=[0,1]
#pylab.plot(dat)
#pylab.ion()
#pylab.draw()
#for i in range (18):
#    dat.append(random.uniform(0,1))
#    pylab.plot(dat)
#    pylab.draw()
#    time.sleep(1)

# ====================================================
# General fenics

# get the values of function u from subspace idx of some mixed function space, V
def dolfinGetDOFmap(V,idx):
    if V.num_sub_spaces() > 1:
        dofmap = V.sub(idx).dofmap().dofs()
    else:
        dofmap = V.dofmap().dofs()
    return dofmap

def dolfinGetFunctionValues(u,V,idx):
    dofmap = dolfinGetDOFmap(V,idx)
    return u.vector().get_local()[dofmap]

def dolfinGetFunctionValuesAtPoint(u,idx,coord):
    return u(coord)[idx]

def dolfinSetFunctionValues(u,unew,V,idx):
    # unew can be a scalar (all dofs will be set to that value), a numpy array, or a list
    dofmap = dolfinGetDOFmap(V,idx)
    u.vector()[dofmap] = unew

def dolfinGetFunctionStats(u,V,idx):
    uvalues = dolfinGetFunctionValues(u,V,idx)
    return {'mean': uvalues.mean(), 'min': uvalues.min(), 'max': uvalues.max(), 'std': uvalues.std()}

def dolfinFindClosestPoint(mesh, coords):
    """
    Given some point and a mesh, returns the coordinates of the nearest vertex
    """
    p = d.Point(coords)
    L = list(d.vertices(mesh))
    distToVerts = [np.linalg.norm(p.array() - x.midpoint().array()) for x in L]
    minDist = min(distToVerts)
    minIdx = distToVerts.index(minDist) # returns the local index (wrt the cell) of the closest point


    closestPoint = L[minIdx].midpoint().array()
    print('Closest point found was %f distance away' % minDist)
    return closestPoint


