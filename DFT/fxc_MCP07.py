import numpy as np

from constants import pi
from DFT.asymptotics import get_fxc_pars, gexc2
from DFT.fxc_GKI import fxc_GKI_real_freq

def fxc_MCP07_static(q, dv, param = 'PZ81', kernel_only = False):

    cfac = 4*pi/dv.kF**2
    q2 = q**2

    CA, CB, CC = get_fxc_pars(dv, param = param)

    fxc_ALDA = -cfac*CA
    akn = CA/(dv.kF**2 * CB)

    # The gradient term
    cxc = gexc2(dv.rs)
    dd = 2.0*cxc/(dv.n**(4.0/3.0)*(4.0*pi*CB)) - 0.5*akn**2

    qeps = 1.e-6
    fxcmcp07_taylor = fxc_ALDA + q2 * ( 2*cxc/dv.n**(4./3.) + \
        (fxc_ALDA*(akn**2/6. + dd) - cfac*CC*akn**2)*q2 )

    def _fxc_MCP07(_q2):
        zp = akn*_q2
        grad = 1.0 + dd*_q2*_q2
        vc = 4.0*pi/_q2
        cl = vc*CB
        cutdown = 1.0 + 1.0/(akn*_q2)**2
        return cl*(np.exp(-zp)*grad - 1.0) - cfac*CC/cutdown


    if hasattr(q,'__len__'):

        fxcmcp07 = np.zeros_like(q)

        qm = q < qeps
        fxcmcp07[qm] = fxcmcp07_taylor[qm]

        qm = q >= qeps
        fxcmcp07[qm] = _fxc_MCP07(q2[qm])

    else:
        if q < qeps:
            fxcmcp07 = fxcmcp07_taylor
        else:
            fxcmcp07 = _fxc_MCP07(q2)

    if kernel_only:
        return fxcmcp07

    return fxcmcp07, fxc_ALDA, akn

def fxc_MCP07(q,omega,dv):

    fxc_omega = fxc_GKI_real_freq(omega, dv, param = 'PZ81', revised = False)
    fxc_q, fxc_ALDA, ikscr = fxc_MCP07_static(q, dv, param = 'PZ81')
    fxc = (1. + np.exp(-ikscr*q**2)*(fxc_omega/fxc_ALDA - 1.) )*fxc_q

    return fxc

def fxc_rMCP07(q,omega,dv):
    
    FITPS = {'A': 3.846991, 'B': 0.471351, 'C': 4.346063, 'D': 0.881313}
    kscr = dv.kF*(FITPS['A'] + FITPS['B']*dv.kF**(1.5))/(1. + dv.kF**2)

    sclfun = (dv.rs/FITPS['C'])**2

    pscl = sclfun + (1. - sclfun)*np.exp(-FITPS['D']*(q/kscr)**2)
    fxc_omega = fxc_GKI_real_freq(omega*pscl, dv, param = 'PW92', revised = True)
    fxc_q, fxc_ALDA, _ = fxc_MCP07_static(q, dv, param = 'PW92')

    fxc = (1. + np.exp(-(q/kscr)**2)*(fxc_omega/fxc_ALDA - 1.))*fxc_q

    return fxc