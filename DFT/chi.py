import numpy as np
from scipy.interpolate import splrep

from constants import pi
from DFT.common import HEG

from DFT.fxc_ALDA import fxc_ALDA
from DFT.fxc_AKCK import fxc_AKCK, fxc_AKCK_dyn
from DFT.fxc_CDOP import fxc_CDOP
from DFT.fxc_GKI import fxc_GKI_real_freq
from DFT.fxc_MCP07 import fxc_MCP07_static, fxc_MCP07, fxc_rMCP07
from DFT.fxc_QV import fxc_QV, fxc_QV_spline, fxcl_static

def WCOUL(q, omega, rs, fxc_param, _DF, fxc_opts = {}):
    return VCOUL(q)*inverse_DF(q, omega, rs, fxc_param, _DF, fxc_opts = fxc_opts)

def inverse_DF(q, omega, rs, fxc_param, _DF, fxc_opts = {}):
    chi = chi_HEG(q, omega, rs, fxc_param, fxc_opts = fxc_opts)

    if _DF == 'TCTC':
        enh = VCOUL(q)
    elif _DF == 'TCTE':
        enh = get_fHxc(q, omega, HEG(rs), fxc_param, fxc_opts = fxc_opts)

    return 1. + enh*chi
   
def spectral_HEG(q, omega, rs, fxc_param, fxc_opts = {}):
    chi = chi_HEG(q, omega, rs, fxc_param, fxc_opts = fxc_opts)
    _HEG = HEG(rs)
    return -chi.imag/(pi*_HEG.n)

def chi_HEG(q, omega, rs, fxc_param, fxc_opts = {}):
    DV = HEG(rs)
    chi_0 = chi_0_HEG(q, omega, DV)
    fHxc = get_fHxc(q, omega, DV, fxc_param, fxc_opts = fxc_opts)
    chi = chi_0/(1. - fHxc*chi_0)
    return chi

def get_fHxc(q, omega, dv, fxc_param, fxc_opts = {}):
    return VCOUL(q) + get_fxc(q, omega, dv, fxc_param, fxc_opts = fxc_opts)

def get_fxc(q, omega, dv, fxc_param, fxc_opts = {}):

    # local in space, local in time
    if fxc_param == 'RPA':
        fxc = 0.
    elif fxc_param == 'ALDA':
        fxc = fxc_ALDA(dv, x_only = False, param = 'PW92')
    elif fxc_param == 'ALDA-shear':
        fxc = fxcl_static(dv)

    # nonlocal in space, local in time
    elif fxc_param == 'AKCK':
        fxc = fxc_AKCK(q,dv)
    elif fxc_param == 'CDOP':
        fxc = fxc_CDOP(q,dv)
    elif fxc_param in ['Static MCP07', 'MCP07-static']:
        fxc = fxc_MCP07_static(q, dv, param = 'PZ81', kernel_only = True)
    
    # local in space, nonlocal in time
    elif fxc_param == 'GKI':
        fxc = fxc_GKI_real_freq(omega, dv, param = 'PW92', revised = True)
    elif fxc_param == 'QV':
        fxc = fxc_QV(omega, dv)
    elif fxc_param == 'QV spline':
        fxc = fxc_QV_spline(fxc_opts).eval(omega)
    
    # nonlocal in space, nonlocal in time
    elif fxc_param == 'MCP07':
        fxc = fxc_MCP07(q, omega, dv)
    elif fxc_param == 'rMCP07':
        fxc = fxc_rMCP07(q, omega, dv)
    elif fxc_param == 'AKCK dynamic':
        fxc = fxc_AKCK_dyn(q,omega,dv)

    return fxc

def VCOUL(q):
    return 4.*pi/q**2

def chi_0_HEG(q, omega, dv):

    z = q/(2.*dv.kF)
    uu = omega/(4.*z*dv.epsF)

    """
        Eq. 3.6 of Lindhard's paper
    """

    zu1 = z - uu + 0.0j
    zu2 = z + uu + 0.0j

    fx = 0.5 + 0.0j
    fx += (1.0-zu1**2)/(8.0*z)*np.log((zu1 + 1.0)/(zu1 - 1.0))
    fx += (1.0-zu2**2)/(8.0*z)*np.log((zu2 + 1.0)/(zu2 - 1.0))

    return -dv.kF*fx/pi**2