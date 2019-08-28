import dolfin as d


# TODO: probably best to make this a class with methods e.g. (forward step)
def picard_loop(cdf,pdf,u,t,dt,T,dT,data):
    t += dt
    T.assign(t)
    dT.assign(dt)

    updateTimeDependentParameters(pdf, t)

    minDim = cdf.dimensionality.min()
    pidx = 0
    compNames = ['cyto', 'pm']
    skipAssembly = False

    while True:
        pidx += 1
        for compName in compNames:#u.keys():
            comp = cdf.loc[cdf.compartment_name==compName].squeeze()
            # solve
            data.timer.start()
            numIter=5 if comp.dimensionality == minDim else 1 # extra steps
            for idx in range(numIter):
                if skipAssembly and compName=='cyto':
                    d.solve(A,u['cyto']['u'].vector(),b,'cg','ilu')
                    b = d.assemble(comp.L)
                else:
                    d.solve(comp.a==comp.L, u[compName]['u'],
                        solver_parameters=data.model_parameters['solver']['dolfin_params'])
                # compute error
                data.computeError(u,compName,data.model_parameters['solver']['norms'])
                # assign new solution to temp variable
                u[compName]['k'].assign(u[compName]['u'])
                if comp.dimensionality > minDim:
                    u[compName]['b'].interpolate(u[compName]['k'])

            print('Component %s solved for in %f seconds'%(compName,data.timer.stop()))

        # check convergence
        #isConvergedRel=[] TODO: implement relative norm convergence conditions
        isConvergedAbs=[]
        abs_err_dict = {}
        for compName in compNames:#u.keys():
            for norm in data.model_parameters['solver']['norms']:
                abs_err = data.errors[compName][norm][-1]
                print('%s norm (%s) : %f ' % (norm, compName, abs_err))
                isConverged = abs_err < data.model_parameters['solver']['linear_abstol']
                abs_err_dict[compName] = abs_err
                isConvergedAbs.append(isConverged)
            if compName=='pm' and abs_err<1e-7 and not skipAssembly: # TODO: make this more robust
                print('SKIPPING ASSEMBLY from now on')
                acyto = cdf.loc[cdf.compartment_name=='cyto','a'].squeeze()
                Lcyto = cdf.loc[cdf.compartment_name=='cyto','L'].squeeze()
                A=d.assemble(acyto)
                b=d.assemble(Lcyto)
                skipAssembly = True


        # exit picard loop if convergence criteria are met
        if all(isConvergedAbs):
            print('Change in norm less than tolerance (%f). Converged in %d picard iterations.'
                % (data.model_parameters['solver']['linear_abstol'], pidx))
            # increase time-step if convergence was quick
            if pidx <= data.model_parameters['solver']['min_picard']:
                dt *= data.model_parameters['solver']['dt_increase_factor']
                print('Solution converged in less than "min_picard" iterations. Increasing time-step by a factor of %f [dt=%f]'
                    % (data.model_parameters['solver']['dt_increase_factor'], dt))

            break

        # decrease time-step if not converging
        isErrorTooHigh = [err>100 for err in list(abs_err_dict.values())]
        if pidx >= data.model_parameters['solver']['max_picard'] or any(isErrorTooHigh):
            # reset
            t -= dt
            dt *= data.model_parameters['solver']['dt_decrease_factor']
            t += dt
            T.assign(t); dT.assign(dt)
            updateTimeDependentParameters(pdf, t)
            
            print('Maximum number of picard iterations reached. Decreasing time-step by a factor of %f [dt=%f]'
                % (data.model_parameters['solver']['dt_decrease_factor'], dt))
            for compName in compNames:#u.keys():
                comp = cdf.loc[cdf.compartment_name==compName].squeeze()
                u[compName]['k'].assign(u[compName]['n'])
                if comp.dimensionality > minDim:
                    u[compName]['b'].interpolate(u[compName]['n'])
            pidx = 0

    return (t,dt,T,dT)


# def updateTimeDependentParameters(pdf, t): 
#     for idx,param in pdf.iterrows():
#         if param.is_time_dependent:
#             newValue = param.symExpr.subs({'t': t}).evalf()
#             param.dolfinConstant.get().assign(newValue)
#             print('%f assigned to time-dependent parameter %s' % (newValue, param.parameter_name))