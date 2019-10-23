def foo(keys):
    for key in keys:
        u=solns[key]
        uavg=u['mean']
        umax=u['max']
        umin=u['min']
        u2=solns_glu[key]
        u2avg=u2['mean']
        u2max=u2['max']
        u2min=u2['min']

        t=u['tvec']
        t2=u2['tvec']

        plt.plot(t,uavg,'-',linewidth=3,label=key); #plt.plot(t,umax,'r-'); plt.plot(t,umin,'r-');
        plt.plot(t2,u2avg,':',linewidth=3,label=key);# plt.plot(t2,u2max,':'); plt.plot(t2,u2min,':');

    plt.legend()
    plt.show()



def foo(comp_name,keys):
    F = 0
    for key in keys:
        F += model.split_forms[comp_name][key]
    return NonlinearVariationalProblem(F,model.u[comp_name]['u'])


split_forms={}
for comp in comp_list:
    split_forms[comp.name] = {}
    comp_forms = model.Forms.select_by('compartment_name', comp.name)
    for form_type in form_types:
        split_forms[comp.name][form_type] = [f for f in comp_forms if f.form_type==form_type]