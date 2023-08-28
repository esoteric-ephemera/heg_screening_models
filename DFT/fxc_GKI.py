from constants import pi
from DFT.asymptotics import get_fxc_high_freq
from DFT.common import GAMMA_GKI, C_GKI
from DFT.fxc_ALDA import fxc_ALDA

def fxc_GKI_real_freq(omega, dv, param = 'PZ81', revised = False):

    fxc_inf = get_fxc_high_freq(dv, x_only = False, param = param)
    fxc_0 = fxc_ALDA(dv, x_only = False, param = param)

    BFAC = (GAMMA_GKI/C_GKI)**(4./3.) 
    delta_fxc = fxc_inf - fxc_0
    scl = BFAC*delta_fxc**(4./3.)

    var = scl**(0.5) * omega

    """
        Imaginary part from E.K.U. Gross and W. Kohn,
        Phys. Rev. Lett. 55, 2850 (1985),
        https://doi.org/10.1103/PhysRevLett.55.2850,
        and erratum Phys. Rev. Lett. 57, 923 (1986).
    """
    var2 = var*var
    im_fxc = var/(1. + var2)**(5./4.)

    re_fxc_0 = 1./GAMMA_GKI
    if revised:
        CPS = [0.174724,3.224459,2.221196,1.891998]
        CPS.append((re_fxc_0*CPS[0])**(16./7.))
        re_fxc = re_fxc_0*(1. - CPS[0]*var2)/(1. + var2*(CPS[1] + var2*(CPS[2] + var2*(CPS[3] + var2*CPS[4]))))**(7./16.)
    else:
        CP = 0.63
        re_fxc = re_fxc_0*(1. - CP*var2)/(1. + (re_fxc_0*CP)**(4./7.) * var2)**(7./4.)

    return fxc_inf - C_GKI*scl**(3./4.)*(re_fxc + 1.j*im_fxc)