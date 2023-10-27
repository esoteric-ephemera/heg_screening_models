import numpy as np

from constants import pi
from DFT.asymptotics import get_fxc_pars
from DFT.common import HEG

def smooth_step(x,a,b):
    f1 = np.exp(a*b)
    f2 = np.exp(-a*x)
    f = (f1 - 1.)*f2/(1. + (f1 - 2.)*f2)
    return f

def simple_LFF(q,dv,c):

    q2 = (q/dv.kF)**2
    q4 = q2*q2

    CA, CB, CC = get_fxc_pars(dv)

    alpha = c[0] + c[1]*np.exp(-abs(c[2])*dv.rs)

    interp1 = smooth_step(q4/16.,c[3],c[4])

    asymp1 = q2*(CA + alpha*q4)
    asymp2 = CB + CC*q2
    LFF = asymp1*interp1 + asymp2*(1. - interp1)

    return LFF

def fxc_AKCK(q,dv):
    cps = [-0.00451760, 0.0155766, 0.422624, 3.516054, 1.015830]
    return -4.*pi*simple_LFF(q,dv,cps)/q**2
