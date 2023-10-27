
import numpy as np

from constants import pi
from DFT.common import HEG
from DFT.asymptotics_2D import get_g_plus_pars
from DFT.fxc_AKCK import smooth_step

def fxc_DPGT_2D(q, dv: HEG):

    """
    B. Davoudi, M. Polini, G.F. Giuliani, and M.P. Tosi,
    Phys. Rev. B 64, 153101 (2001)
    DOI: 10.1103/PhysRevB.64.153101
    """

    # majik numbor time - Eqs. 11
    rsr = dv.rs/10.

    rs_exp = 0.9218
    alpha_pos = (0.1598 + 0.8931*rsr**rs_exp)/(1. + 0.8793*rsr**rs_exp)
    g2 = 0.5824*rsr**2 - 0.4272*rsr
    g4 = 0.2960*rsr - 1.003*rsr**(5./2.) + 0.9466*rsr**3
    g6 = -0.0585*rsr**2
    g8 = 0.0131*rsr**2

    # just below Eq. 10
    x = q/dv.kF
    x2 = x*x
    p_pos = x2*(g2 + x2*(g4 + x2*(g6 + x2*g8)))

    # Eq. 10
    ersr = np.exp(rsr)
    A, B, C = get_g_plus_pars(dv)
    scale_fac = ersr/(1. + x2*(A*ersr/B)**2)**(0.5) - np.expm1(rsr)*np.exp(-x2/4.)
    g_pos = x*(A*scale_fac - C*np.expm1(-x2)) + p_pos*np.exp(-alpha_pos*x2)

    return -(2*pi)*g_pos/q
