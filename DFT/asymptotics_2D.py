
from constants import pi

from DFT.fxc_ALDA_2D import fxc_ALDA_2D, eps_c_2D_unp
from DFT.common import HEG

def on_top_pair_dist(rs):
    """
    M. Polini, G. Sica, B. Davoudi, and M. P. Tosi, 
    
    J. Phys.: Condens. Matter 13, 3591 (2001).
    DOI: 10.1088/0953-8984/13/15/303
    """
    return 0.5/(1. + 1.372*rs + 0.0830*rs**2)

def get_g_plus_pars(dv : HEG):

    """
    B. Davoudi, M. Polini, G.F. Giuliani, and M.P. Tosi,
    Phys. Rev. B 64, 153101 (2001)
    DOI: 10.1103/PhysRevB.64.153101
    """

    # Eq. 4 
    Apos = -dv.kF*fxc_ALDA_2D(dv.rs)/(2.*pi)

    # Just below Eq. 8
    Bpos = 1. - on_top_pair_dist(dv.rs)

    # Eq. 8, converting Ry --> Hartree
    eps_c, d_eps_c_drs, _ = eps_c_2D_unp(dv.rs)
    Cpos = -dv.rs*(eps_c + dv.rs*d_eps_c_drs)/2**(0.5)

    return Apos, Bpos, Cpos