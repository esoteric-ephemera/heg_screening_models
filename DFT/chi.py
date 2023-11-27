import numpy as np

from constants import pi
from DFT.common import HEG
from DFT.fxc import get_fxc

def WCOUL(q, omega, rs, fxc_param, _DF, fxc_opts = {}, d = 3):
    return VCOUL(q, d = d)*inverse_DF(q, omega, rs, fxc_param, _DF, fxc_opts = fxc_opts, d = d)

def TCTE_DF(q, omega, rs, fxc_param, fxc_opts = {}, d = 3):
    DV = HEG(rs,d=d)
    LF_ENH = get_fHxc(q, omega, DV, fxc_param, fxc_opts = fxc_opts, d=d)
    chi0 = chi_0_HEG(q, omega, DV, d = d)
    return 1. - LF_ENH*chi0

def inverse_DF(q, omega, rs, fxc_param, _DF, fxc_opts = {}, d = 3):
    chi = chi_HEG(q, omega, rs, fxc_param, fxc_opts = fxc_opts, d = d)

    if _DF == 'TCTC':
        enh = VCOUL(q, d = d)
    elif _DF == 'TCTE':
        enh = get_fHxc(q, omega, HEG(rs,d=d), fxc_param, fxc_opts = fxc_opts, d=d)

    return 1. + enh*chi
   
def spectral_HEG(q, omega, rs, fxc_param, fxc_opts = {}, d = 3):
    chi = chi_HEG(q, omega, rs, fxc_param, fxc_opts = fxc_opts, d = d)
    _HEG = HEG(rs,d=d)
    return -chi.imag/(pi*_HEG.n)

def chi_HEG(q, omega, rs, fxc_param, fxc_opts = {}, d = 3):
    DV = HEG(rs,d=d)
    chi_0 = chi_0_HEG(q, omega, DV, d = d)
    fHxc = get_fHxc(q, omega, DV, fxc_param, fxc_opts = fxc_opts, d = d)
    chi = chi_0/(1. - fHxc*chi_0)
    return chi

def get_fHxc(q, omega, dv, fxc_param, fxc_opts = {}, d=3):
    return VCOUL(q,d=d) + get_fxc(q, omega, dv, fxc_param, fxc_opts = fxc_opts, d = d)

def VCOUL(q, d = 3):
    if d == 2:
        vc = 2*pi/np.abs(q)
    elif d == 3:
        vc = 4*pi/q**2
    else:
        raise SystemExit(f"Cannot compute bare Coulomb interaction in {d} dimensions!")
    return vc

def chi_0_HEG(q,omega,dv, d = 3):

    if d == 2:
        return _chi_0_HEG_2D(q, omega, dv)
    elif d == 3:
        return _chi_0_HEG_3D(q, omega, dv)
    else:
        raise SystemExit(f"Cannot compute Lindhard function in {d} dimensions!")

def _chi_0_HEG_3D(q, omega, dv):

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


def heaviside(x):
    theta = np.zeros_like(x)
    xmsk = x > 0.
    theta[xmsk] = 1.
    return theta

def _phi_chi_0_HEG(x):
    arg = np.abs(x)**2 - 1.
    step = heaviside(arg)
    return np.sign(x.real)*step*(step*arg)**(0.5)

def _psi_chi_0_HEG(x):
    arg = 1. - np.abs(x)**2
    step = heaviside(arg)
    return step*(step*arg)**(0.5)

def _chi_0_HEG_2D(q, omega, dv):

    """
    Eqs. 3 of F. Stern, Phys. Rev. Lett. 18, 546-548 (1967)

    DOI: 10.1103/PhysRevLett.18.546
    """
    z = q/(2*dv.kF)
    u = omega/(q*dv.kF)

    prefactor = -1./pi

    re_chi_0 = -(_phi_chi_0_HEG(z - u) + _phi_chi_0_HEG(z + u))
    im_chi_0 = _psi_chi_0_HEG(z - u) - _psi_chi_0_HEG(z + u)
    chi_0 = prefactor*(1. + (re_chi_0 + 1.j*im_chi_0)/(2.*z) )
    return chi_0

def _chi_0_HEG_imfreq(q, omega, dv):

    """  
    Eq. 6 of R. Asgari, M. Polini, B. Davoudi, and M.P. Tosi,

    Phys Rev. B 68, 235116 (2003).
    DOI: 10.1103/PhysRevB.68.235116

    Using this for sanity check of .chi_0_HEG
    """
    z = q/(2.*dv.kF)
    u = omega/(q*dv.kF)
    fzu = z**2 - u**2 - 1.
    chi0_imfreq = (fzu + (fzu**2 + 4*(z*u)**2)**(0.5))**(0.5)/(2**(0.5)*z) - 1.
    return chi0_imfreq/pi