import numpy as np

def bracket(fun,bds,nstep=500,vector=False,args=(),kwargs={}):

    step = (bds[1]-bds[0])/nstep
    ivals = []
    if vector:
        tmpl = np.arange(bds[0],bds[1],step)
        funl = fun(tmpl,*args,**kwargs)
        ofun = funl[0]
        for istep in range(1,nstep):
            if ofun*funl[istep] <= 0:
                ivals.append([tmpl[istep-1],tmpl[istep]])
            ofun = funl[istep]
    else:
        tmp = bds[0]
        for istep in range(nstep):
            cfun = fun(tmp,*args,**kwargs)
            if istep == 0:
                ofun = cfun
            if ofun*cfun <= 0:
                ivals.append([tmp-step,tmp])
            ofun = cfun
            tmp += step
    return ivals
