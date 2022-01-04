"""
Functions to help with managing solutions / post-processing
"""
import dolfin as d
import mpi4py.MPI as pyMPI
import numpy as np
import matplotlib.pyplot as plt
import pickle
from numbers import Number
import os
import petsc4py.PETSc as PETSc
from collections import defaultdict as ddict
Print = PETSc.Sys.Print
import matplotlib.lines as mlines
from stubs.common import round_to_n

comm = d.MPI.comm_world
size = comm.size
rank = comm.rank
root = 0

from stubs import unit as unit
#import networkx as nx
#import stubs.model_assembly as model_assembly

# # matplotlib settings
# lwsmall = 1.5
# lwmed = 3
# lineopacity = 0.6
# fsmed = 7
# fssmall = 5

class Data:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.append_flag = False
        self.solutions = {}
        self.probe_solutions = ddict(list)
        self.fluxes = ddict(list)
        self.tvec=[]
        self.dtvec=[]
        self.nl_idxvec=[]
        self.errors = {}
        self.parameters = ddict(list)
        #self.plots = {'solutions': {'fig': plt.figure(), 'subplots': []}}
        self.plots = {'solutions': plt.figure(), 'solver_status': plt.subplots()[0], 'parameters': plt.figure(), 'fluxes': plt.figure()}
        self.color_list = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']


    def init_solution_files(self, sc, config):
        output_type = config.output_type

        for sp_name, sp in sc.items:
            self.solutions[sp_name] = {}

            self.solutions[sp_name]['num_species'] = sp.compartment.num_species
            self.solutions[sp_name]['comp_name'] = sp.compartment_name
            self.solutions[sp_name]['comp_idx'] = int(sp.dof_index)
            self.solutions[sp_name]['concentration_units'] = sp.concentration_units

            if output_type=='vtk':
                file_str = self.config.directory['solutions'] + '/' + sp_name + '.pvd'
                self.solutions[sp_name][output_type] = d.File(file_str)
            elif output_type=='xdmf':
                file_str = self.config.directory['solutions'] + '/' + sp_name + '.xdmf'
                self.solutions[sp_name][output_type] = d.XDMFFile(comm,file_str)
            elif config.flags['store_solutions']==False or output_type==None:
                self.solutions[sp_name][output_type] = None
            else:
                raise Exception("Unknown solution file type")


    def store_solution_files(self, u, t, config):
        output_type = config.output_type

        if config.flags['store_solutions']==False or output_type==None:
            return
        for sp_name in self.solutions.keys():
            comp_name = self.solutions[sp_name]['comp_name']
            comp_idx = self.solutions[sp_name]['comp_idx']
            #print("spname: %s" % sp_name)

            if self.solutions[sp_name]['num_species'] == 1:
                if output_type=='vtk':
                    self.solutions[sp_name]['vtk'] << (u[comp_name]['u'], t)
                elif output_type=='xdmf':
                    #file_str = self.config.directory['solutions'] + '/' + sp_name + '.xdmf'
                    #with d.XDMFFile(file_str) as xdmf:
                    #    xdmf.write(u[comp_name]['u'], t)
                    self.solutions[sp_name][output_type].write_checkpoint(u[comp_name]['u'], "u", t, append=self.append_flag)
                    self.solutions[sp_name][output_type].close()
                else:
                    raise Exception("Unknown output type")
            else:
                if output_type=='vtk':
                    self.solutions[sp_name]['vtk'] << (u[comp_name]['u'].split()[comp_idx], t)
                elif output_type=='xdmf':
                    # writing xdmf on submeshes fails in parallel
                    # TODO: fix me
                    if comp_name=='cyto':
                        self.solutions[sp_name][output_type].write_checkpoint(u[comp_name]['u'].split()[comp_idx], "u", t, append=self.append_flag)
                        self.solutions[sp_name][output_type].close()
                else:
                    raise Exception("Unknown output type")

        self.append_flag = True # append to xmdf files rather than write over

        Print(f"Solutions dumped into {output_type}.")


    def compute_function_stats(self, sp):
        """
        Computes and returns min, mean, and max values of a species
        """
        uvec    = self.model.dolfin_get_function_values(sp)
        dx      = sp.compartment.dx
        u       = self.model.u[sp.compartment_name]['u']
        if self.model.V[sp.compartment_name].num_sub_spaces() == 0:
            usub = u
        else:
            usub = u.sub(sp.dof_index)

        umean   = d.assemble(usub*dx) / d.assemble(1*dx)
        umin    = uvec.min()
        umax    = uvec.max()
        ustd    = uvec.std()
        return {'min': umin, 'mean': umean, 'max': umax, 'std': ustd}


    def compute_statistics(self, u, t, dt, sc, pc, cc, fc, nl_idx):
        #for sp_name in speciesList:
        for sp_name, sp in sc.items:
            comp_name = self.solutions[sp_name]['comp_name']
            comp_idx = self.solutions[sp_name]['comp_idx']

            # compute statistics and append values
            ustats = self.compute_function_stats(sp)
            #ustats = dolfin_get_function_stats(sp)
            for key, value in ustats.items():
                if key not in self.solutions[sp_name].keys():
                    self.solutions[sp_name][key] = [value]
                else:
                    self.solutions[sp_name][key].append(value)

        # store time dependent parameters
        for param in pc:
            if param.is_time_dependent:
                self.parameters[param.name].append(param.value)

        # store fluxes
        flux_names = [flux_name for flux_name, flux in fc.items if flux.track_value]
        # remove forward/reverse flux labels
        temp = [flux_name.replace(' (r)','') for flux_name in flux_names]
        temp = [flux_name.replace(' (f)','') for flux_name in temp]

        flux_indices = set() # indices of fluxes we dont want to sum
        summed_flux_indices = set() # indices of fluxes we want to sum (in terms of tuples)
        for i in range(len(temp)-1):
            for j in range(i+1,len(temp)):
                if temp[i] == temp[j]:
                    summed_flux_indices.add((i,j))

        flatten = lambda tuple_set: set([item for sublist in [list(i) for i in tuple_set] for item in sublist]) # flattens a set of tuples into a set
        flux_indices = set(range(len(temp)))
        flux_indices = flux_indices.difference(flatten(summed_flux_indices))

        # compute (assemble) fluxes
        for i in flux_indices:
            flux_name = flux_names[i]
            flux = fc[flux_name]
            area_units = cc[flux.source_compartment].compartment_units**2
            scale_to_molecule_per_s = (1*flux_units*area_units).to(unit.molecule/unit.s).magnitude
            #value = sum(d.assemble(flux.dolfin_flux))*scale_to_molecule_per_s
            value = d.assemble(flux.dolfin_flux)*scale_to_molecule_per_s
            self.fluxes[flux_name].append(value)

        # compute (assemble) sums of fluxes
        for i,j in summed_flux_indices:
            flux_name_1 = flux_names[i]
            flux_name_2 = flux_names[j]
            flux_1 = fc[flux_name_1]
            flux_2 = fc[flux_name_2]
            area_units_1 = cc[flux_1.source_compartment].compartment_units**2
            area_units_2 = cc[flux_2.source_compartment].compartment_units**2

            scale_to_molecule_per_s_1 = (1*flux_1.flux_units*area_units_1).to(unit.molecule/unit.s).magnitude
            scale_to_molecule_per_s_2 = (1*flux_2.flux_units*area_units_2).to(unit.molecule/unit.s).magnitude
            new_flux_name = temp[i] + ' (SUM)'

#            value = sum(d.assemble(flux_1.dolfin_flux))*scale_to_molecule_per_s_1 \
#                    + sum(d.assemble(flux_2.dolfin_flux))*scale_to_molecule_per_s_2
            value = d.assemble(flux_1.dolfin_flux)*scale_to_molecule_per_s_1 \
                    + d.assemble(flux_2.dolfin_flux)*scale_to_molecule_per_s_2

            self.fluxes[new_flux_name].append(value)

        self.tvec.append(t)
        self.dtvec.append(dt)
        if len(nl_idx.values()) == 0:
            self.nl_idxvec.append(0)
        else:
            self.nl_idxvec.append(max(nl_idx.values()))

    #def computeProbeValues(self, u, t, dt, sc, pc, cc, fc, nl_idx):
    def compute_probe_values(self, u, sc):
        """
        Computes the values of functions at various coordinates
        """
        #x_list = self.config.output['points_x']
        #y_list = self.config.output['points_y']

        for sp_name, coord_list in self.config.probe_plot.items():
            comp = sc[sp_name].compartment
            sp_idx = sc[sp_name].dof_index
            if sp_name not in self.probe_solutions.keys():
                self.probe_solutions[sp_name] = {}
            for coords in coord_list:
                u_coords = u[comp.name]['u'](coords)
                if coords not in self.probe_solutions[sp_name].keys():
                    self.probe_solutions[sp_name][coords] = []

                if comp.num_species > 1:
                    u_eval = u_coords[sp_idx]
                else:
                    u_eval = u_coords

                self.probe_solutions[sp_name][coords].append(u_eval)


    def compute_error(self, u, comp_name, errorNormKey):
        errorNormDict = {'L2': 2, 'Linf': np.Inf}
        if comp_name not in self.errors.keys():
            self.errors[comp_name] = {}
        #for key in errorNormKeys:
        error_norm = errorNormDict[errorNormKey]
        u_u = u[comp_name]['u'].vector().get_local()
        u_k = u[comp_name]['k'].vector().get_local()
        u_n = u[comp_name]['n'].vector().get_local()
        abs_err = np.linalg.norm(u_u - u_k, ord=error_norm)
        #rel_err = np.linalg.norm((u_u - u_k)/u_n, ord=error_norm)
        #rel_err = 1.0
       
        if errorNormKey not in self.errors[comp_name].keys():
            self.errors[comp_name][errorNormKey] = {'abs': []}
        #     self.errors[comp_name][errorNormKey]['abs'] = [abs_err]
        #     self.errors[comp_name][errorNormKey]['rel'] = [rel_err]
        # else:
        #     self.errors[comp_name][errorNormKey]['abs'].append(abs_err)
        #     self.errors[comp_name][errorNormKey]['rel'].append(rel_err)

        #self.errors[comp_name][errorNormKey]['rel'].append(rel_err)
        self.errors[comp_name][errorNormKey]['abs'].append(abs_err)
        Print("Absolute error [%s] in the %s norm: %f" %(comp_name, errorNormKey, abs_err))

        return abs_err

#            self.errors[comp_name][errorNormKey].append(np.linalg.norm(u[comp_name]['u'].vector().get_local()
#                                    - u[comp_name]['k'].vector().get_local(), ord=error_norm))

    def init_plot(self, config, sc, fc):
        if rank==root:
            if not os.path.exists(config.directory['plots']):
                os.makedirs(config.directory['plots'])
        Print("Created directory %s to store plots" % config.directory['plots'])

        maxCols = 3
        # solution plots 
        self.groups = list(set([p.group for p in sc]))
        if 'Null' in self.groups: self.groups.remove('Null')
        numPlots = len(self.groups)
        #numPlots = len(self.solutions.keys())
        subplotCols = min([maxCols, numPlots])
        subplotRows = int(np.ceil(numPlots/subplotCols))
        for idx in range(numPlots):
            self.plots['solutions'].add_subplot(subplotRows,subplotCols,idx+1)

        # parameter plots
        numPlots = len(self.parameters.keys())
        if numPlots > 0:
            subplotCols = min([maxCols, numPlots])
            subplotRows = int(np.ceil(numPlots/subplotCols))
            for idx in range(numPlots):
                self.plots['parameters'].add_subplot(subplotRows,subplotCols,idx+1)

        # flux plots
        flux_names = [flux_name for flux_name, flux in fc.items if flux.track_value]
        numPlots = len(flux_names)
        if numPlots > 0:
            # remove forward/reverse flux labels
            temp = [flux_name.replace(' (r)','') for flux_name in flux_names]
            temp = [flux_name.replace(' (f)','') for flux_name in temp]
            for i in range(len(temp)-1):
                for j in range(i+1,len(temp)):
                    if temp[i] == temp[j]:
                        numPlots -= 1

            subplotCols = min([maxCols, numPlots])
            subplotRows = int(np.ceil(numPlots/subplotCols))
            if numPlots > 0:
                for idx in range(numPlots):
                    self.plots['fluxes'].add_subplot(subplotRows,subplotCols,idx+1)

    def plot_parameters(self, config, figsize=(120,40)):
        """
        Plots time dependent parameters
        """
        plot_settings = config.plot_settings
        dir_settings = config.directory
        if len(self.parameters.keys()) > 0:
            for idx, key in enumerate(sorted(self.parameters.keys())):
                param = self.parameters[key]
                subplot = self.plots['parameters'].get_axes()[idx]
                subplot.clear()
                subplot.plot(self.tvec, param, linewidth=plot_settings['linewidth_small'], color='b')

                subplot = self.plots['parameters'].get_axes()[idx]
                subplot.title.set_text(key)
                subplot.title.set_fontsize(plot_settings['fontsize_med'])
            for ax in self.plots['parameters'].axes:
                ax.ticklabel_format(useOffset=False)
                plt.setp(ax.get_xticklabels(), fontsize=plot_settings['fontsize_small'])
                plt.setp(ax.get_yticklabels(), fontsize=plot_settings['fontsize_small'])
                ax.yaxis.get_offset_text().set_fontsize(fontsize=plot_settings['fontsize_small'])

            self.plots['parameters'].tight_layout()
            #plt.tight_layout()
            self.plots['parameters'].savefig(dir_settings['plots']+'/'+plot_settings['figname']+'_params', figsize=figsize,dpi=300)#,bbox_inches='tight')

    def plot_fluxes(self, config, figsize=(120,120)):
        """
        Plots assembled fluxes
        Note: assemble(flux) (measure is a surface) [=]
        (vol_concentration*length/s)*length^2 * SCALING FACTOR [length^3*molecule/vol_concentration] -> molecules/s
        """
        plot_settings = config.plot_settings
        dir_settings = config.directory

        if len(self.fluxes.keys()) > 0:
            #for idx, (flux_name,flux) in enumerate(self.fluxes.items()):
            for idx, flux_name in enumerate(sorted(self.fluxes.keys())):
                flux = self.fluxes[flux_name]
                subplot = self.plots['fluxes'].get_axes()[idx]
                subplot.clear()
                subplot.plot(self.tvec, flux, linewidth=plot_settings['linewidth_med'], color='b')

                subplot = self.plots['fluxes'].get_axes()[idx]
                subplot.title.set_text(flux_name)
                subplot.title.set_fontsize(plot_settings['fontsize_med'])
            for ax in self.plots['fluxes'].axes:
                ax.ticklabel_format(useOffset=False)
                plt.setp(ax.get_xticklabels(), fontsize=plot_settings['fontsize_small'])
                plt.setp(ax.get_yticklabels(), fontsize=plot_settings['fontsize_small'])
                ax.yaxis.get_offset_text().set_fontsize(fontsize=plot_settings['fontsize_small'])
            
            self.plots['fluxes'].suptitle('Fluxes [molecules/s]', fontsize=plot_settings['fontsize_med'])
            self.plots['fluxes'].tight_layout()
            #plt.tight_layout()
            self.plots['fluxes'].savefig(dir_settings['plots']+'/'+plot_settings['figname']+'_fluxes', figsize=figsize,dpi=300)#,bbox_inches='tight')


    def plot_solutions(self, config, sc, figsize=(160,160)):
        plot_settings = config.plot_settings
        dir_settings = config.directory

        # plot solutions together by group

        for idx, group in enumerate(sorted(self.groups)):
            subplot = self.plots['solutions'].get_axes()[idx]
            subplot.clear()
            sidx = 0
            for param_name, param in sc.items:
                if param.group == group:

                    soln =  self.solutions[param_name]
                    subplot.plot(self.tvec, soln['min'], linewidth=plot_settings['linewidth_small']*0.5, color=self.color_list[sidx])
                    subplot.plot(self.tvec, soln['mean'], linewidth=plot_settings['linewidth_med'], color=self.color_list[sidx], label=param_name)
                    subplot.plot(self.tvec, soln['max'], linewidth=plot_settings['linewidth_small']*0.5, color=self.color_list[sidx])

                    unitStr = '{:P}'.format(self.solutions[param_name]['concentration_units'].units)
                    sidx += 1
            subplot = self.plots['solutions'].get_axes()[idx]
            subplot.legend(fontsize=plot_settings['fontsize_small'])
            subplot.title.set_text(group)# + ' [' + unitStr + ']')
            subplot.title.set_fontsize(plot_settings['fontsize_med'])


        #self.plots['solutions'].tight_layout()
        for ax in self.plots['solutions'].axes:
            ax.ticklabel_format(useOffset=False)
            plt.setp(ax.get_xticklabels(), fontsize=plot_settings['fontsize_small'])
            plt.setp(ax.get_yticklabels(), fontsize=plot_settings['fontsize_small'])
            ax.yaxis.get_offset_text().set_fontsize(fontsize=plot_settings['fontsize_small'])

        self.plots['solutions'].tight_layout()
        #plt.tight_layout()
        self.plots['solutions'].savefig(dir_settings['plots']+'/'+plot_settings['figname'], figsize=figsize,dpi=300)#,bbox_inches='tight')
        self.plots['solutions'].savefig(dir_settings['plots']+'/'+plot_settings['figname']+'.svg', format='svg', figsize=figsize,dpi=300)#,bbox_inches='tight')


    def plot_solver_status(self, config, figsize=(85,40)):
        plot_settings = config.plot_settings
        dir_settings = config.directory
        nticks=14
        nround=3

        plt.close(self.plots['solver_status'])
        self.plots['solver_status'] = plt.subplots()[0]#.clear()

        if len(self.plots['solver_status'].axes) == 1:
            ax2 = self.plots['solver_status'].axes[0].twinx()
            ax3 = self.plots['solver_status'].axes[0].twiny()
        axes = self.plots['solver_status'].axes

        axes[0].set_ylabel('$\Delta$t [ms]', fontsize=plot_settings['fontsize_med']*2, color='blue')
        axes[0].plot([dt*1000 for dt in self.dtvec], color='blue')
        dt_ticks = np.geomspace(min(self.dtvec), max(self.dtvec), nticks)
        dt_ticks = [round_to_n(dt*1000,nround) for dt in dt_ticks]
        axes[0].set_yscale('log')
        axes[0].set_xlabel('Solver Iteration', fontsize=plot_settings['fontsize_med']*2)
        axes[0].set_yticks(dt_ticks)
        axes[0].set_yticklabels([str(dt) for dt in dt_ticks], fontsize=plot_settings['fontsize_small']*2)
        axes[1].set_ylabel('Newton iterations', fontsize=plot_settings['fontsize_med']*2, color='orange')
        axes[1].plot(self.nl_idxvec, color='orange')
        axes[1].tick_params(labelsize=plot_settings['fontsize_small']*2)
        axes[2].set_xlabel('Time [ms]', fontsize=plot_settings['fontsize_med']*2)
        indices = [int(x) for x in np.linspace(0,len(self.tvec)-1,nticks)]
        axes[0].set_xticks(indices)
        axes[0].set_xticklabels([str(idx) for idx in indices], fontsize=plot_settings['fontsize_small']*2)
        time_ticks = [np.around(self.tvec[idx]*1000,1) for idx in indices]
        #axes[2].clear()
        axes[2].set_xticks(indices)
        axes[2].set_xticklabels([str(t) for t in time_ticks], fontsize=plot_settings['fontsize_small']*2)

        plt.minorticks_off()
        plt.tight_layout()
        self.plots['solver_status'].savefig(dir_settings['plots']+'/'+plot_settings['figname']+'_solver', figsize=figsize,dpi=300)


    def output_pickle(self):
        """
        Outputs solution statistics as a serialized pickle
        """
        newDict = {}

        # statistics over entire domain
        saveKeys = ['min', 'max', 'mean', 'std']
        for sp_name in self.solutions.keys():
            newDict[sp_name] = {}
            #for key in self.solutions[sp_name].keys():
            for key in saveKeys:
                newDict[sp_name][key] = self.solutions[sp_name][key]
        newDict['tvec'] = self.tvec

        # fluxes
        for flux_name in self.fluxes.keys():
            newDict[flux_name] = self.fluxes[flux_name]

        # solutions at specific coordinates
        for sp_name in self.probe_solutions.keys():
            for coord, val in self.probe_solutions[sp_name].items():
                newDict[sp_name][coord] = val

        # pickle file
        stats_dir = self.config.directory['solutions']+'/stats'
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)
        with open(stats_dir+'/'+'pickled_solutions.obj', 'wb') as pickle_file:
            pickle.dump(newDict, pickle_file)

        Print('Solutions dumped into pickle.')

    def output_csv(self):
        """
        Outputs solution statistics as a csv
        """
        stats_dir = self.config.directory['solutions']+'/stats'
        probe_dir = self.config.directory['solutions']+'/probe'
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)
        if not os.path.exists(probe_dir):
            os.makedirs(probe_dir)

        # statistics over entire domain
        saveKeys = ['min', 'max', 'mean', 'std']
        for sp_name in self.solutions.keys():
            #for key in self.solutions[sp_name].keys():
            for key in saveKeys:
                np.savetxt(stats_dir+'/'+sp_name+'_'+str(key)+'.csv', 
                           self.solutions[sp_name][key], delimiter=',')
        np.savetxt(stats_dir+'/'+'tvec'+'.csv', self.tvec, delimiter=',')

        # fluxes
        for flux_name in self.fluxes.keys():
            np.savetxt(stats_dir+'/'+flux_name+'.csv', 
                       self.fluxes[flux_name], delimiter=',')

        # solutions at specific coordinates
        for sp_name in self.probe_solutions.keys():
            for coord, val in self.probe_solutions[sp_name].items():
                if len(coord) == 2:
                    coord_str = "({coord[0]:.3f}_{coord[1]:.3f})".replace('.',',')
                elif len(coord) == 3:
                    coord_str = "({coord[0]:.3f}_{coord[1]:.3f}_{coord[2]:.3f})".replace('.',',')
                else: 
                    raise Exception("Coordinates must be in two or three dimensions.")

                coord_str = str(coord).replace(', ','_').replace('.',',')
                np.savetxt(probe_dir+'/'+sp_name+'_'+coord_str+'.csv', 
                           val, delimiter=',')

        Print('Solutions dumped into CSV.')



# ====================================================
# General fenics

# # get the values of function u from subspace idx of some mixed function space, V
# def dolfin_get_dof_indices(V,species_idx):
#     """
#     Returned indices are *local* to the CPU (not global)
#     function values can be returned e.g.
#     indices = dolfin_get_dof_indices(V,species_idx)
#     u.vector().get_local()[indices]
#     """
#     if V.num_sub_spaces() > 1:
#         indices = np.array(V.sub(species_idx).dofmap().dofs())
#     else:
#         indices = np.array(V.dofmap().dofs())
#     first_idx, last_idx = V.dofmap().ownership_range() # indices that this CPU owns

#     return indices-first_idx # subtract index offset to go from global -> local indices


# def reduceVector(u):
#     """
#     comm.allreduce() only works when the incoming vectors all have the same length. We use comm.Gatherv() to gather vectors
#     with different lengths
#     """
#     sendcounts = np.array(comm.gather(len(u), root)) # length of vectors being sent by workers
#     if rank == root:
#         print("reduceVector(): CPUs sent me %s length vectors, total length: %d"%(sendcounts, sum(sendcounts)))
#         recvbuf = np.empty(sum(sendcounts), dtype=float)
#     else:
#         recvbuf = None

#     comm.Gatherv(sendbuf=u, recvbuf=(recvbuf, sendcounts), root=root)

#     return recvbuf



# def dolfinGetFunctionValues(u,species_idx):
#     """
#     Returns the values of a VectorFunction. When run in parallel this will *not* double-count overlapping vertices
#     which are shared by multiple CPUs (a simple call to u.sub(species_idx).compute_vertex_values() will do this though...)
#     """
#     V = u.function_space()
#     indices = dolfin_get_dof_indices(V,species_idx)
#     uvec = u.vector().get_local()[indices]
#     return uvec

# # when run in parallel, some vertices are shared by multiple CPUs. This function will return all the function values on
# # the invoking CPU. Note that summing/concatenating the results from this function over multiple CPUs will result in
# # double-counting of the overlapping vertices!
# #def dolfinGetFunctionValuesParallel(u,idx):
# #    uvec = u.sub(idx).compute_vertex_values()
# #    return uvec


# def dolfinGetFunctionValuesAtPoint(u, coord, species_idx=None):
    
#     Returns the values of a dolfin function at the specified coordinate 
#     :param dolfin.function.function.Function u: Function to extract values from
#     :param tuple coord: tuple of floats indicating where in space to evaluate u e.g. (x,y,z)
#     :param int species_idx: index of species
#     :return: list of values at point. If species_idx is not specified it will return all values
    
#     if species_idx is not None:
#         return u(coord)[species_idx]
#     else:
#         return u(coord)

# # def dolfinSetFunctionValues(u,unew,V,idx):
# #     # unew can be a scalar (all dofs will be set to that value), a numpy array, or a list
# #     dofmap = dolfinGetDOFmap(V,idx)
# #     u.vector()[dofmap] = unew


# def dolfinSetFunctionValues(u,unew,species_idx):
#     """
#     unew can either be a scalar or a vector with the same length as u
#     """
#     V = u.function_space()

#     indices = dolfin_get_dof_indices(V, species_idx)
#     uvec = u.vector()
#     values = uvec.get_local()
#     values[indices] = unew

#     uvec.set_local(values)
#     uvec.apply('insert')


# # def dolfinGetFunctionStats(u,species_idx):
# #     V = u.function_space()
# #     uvalues = dolfinGetFunctionValues(u,species_idx)
# #     return {'mean': uvalues.mean(), 'min': uvalues.min(), 'max': uvalues.max(), 'std': uvalues.std()}

# # def dolfin_get_function_stats(sp):
# #     dx = sp.compartment.dx
# #     umean = d.assemble(sp.u['u'] * dx) / d.assemble(1 * dx)
#     # umin = sp.


# def dolfinFindClosestPoint(mesh, coords):
#     """
#     Given some point and a mesh, returns the coordinates of the nearest vertex
#     """
#     p = d.Point(coords)
#     L = list(d.vertices(mesh))
#     distToVerts = [np.linalg.norm(p.array() - x.midpoint().array()) for x in L]
#     minDist = min(distToVerts)
#     minIdx = distToVerts.index(minDist) # returns the local index (wrt the cell) of the closest point

#     closestPoint = L[minIdx].midpoint().array()

#     if size > 1:
#         min_dist_global, min_idx = comm.allreduce((minDist,rank), op=pyMPI.MINLOC)

#         if rank == min_idx:
#             comm.Send(closestPoint, dest=root)

#         if rank == root:
#             comm.Recv(closestPoint, min_idx)
#             print("CPU with rank %d has the closest point to %s: %s. The distance is %s" % (min_idx, coords, closestPoint, min_dist_global))
#             return closestPoint, min_dist_global
#     else:
#         return closestPoint, minDist


