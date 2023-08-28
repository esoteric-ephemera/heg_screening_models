import numpy as np
from constants import pi

from DFT.fxc_ALDA import fxc_ALDA, LDA_derivs
from DFT.common import HEG

gex2 = -5./(216.*pi*(3.*pi**2)**(1./3.))

def gec2(rs):
    beta0 = 0.066725
    ca = 3.0
    cb = 1.046
    cc = 0.100
    cd = 1.778*cc
    beta_acgga = beta0*(1 + ca*rs*(cb + cc*rs))/(1 +  ca*rs*(1. + cd*rs))
    return beta_acgga/16.*(pi/3.)**(1./3.)

def gexc2(rs):
    return gex2 + gec2(rs)

def Bpos(dv : HEG):
    a1 = 2.15
    a2 = 0.435
    b1 = 1.57
    b2 = 0.409
    B = (1. + dv.rsh*(a1 + a2*dv.rs))/(3. + dv.rsh*(b1 + b2*dv.rs))
    return B

def get_fxc_pars(dv : HEG, x_only = False, param = 'PW92'):

    eps_c, d_eps_c_d_rs = LDA_derivs(dv, param = param)

    fxc_alda = fxc_ALDA(dv, x_only = x_only, param = param)
    Apos = -dv.kF**2*fxc_alda/(4.*pi)

    d_rs_ec_drs = eps_c + dv.rs*d_eps_c_d_rs
    C = -pi*d_rs_ec_drs/(2.*dv.kF)

    return Apos, Bpos(dv), C

def get_fxc_high_freq(dv, x_only = False, param = 'PW92'):

    """
     from Iwamato and Gross, Phys. Rev. B 35, 3003 (1987),
     f(q,omega=infinity) = -4/5 n^(2/3)*d/dn[ eps_xc/n^(2/3)] + 6 n^(1/3) + d/dn[ eps_xc/n^(1/3)]
     eps_xc is XC energy per electron
    """

    # exchange contribution is -1/5 [3/(pi*n^2)]^(1/3)
    finf_x = -1.0/(5.0)*(3.0/(pi*dv.n**2))**(1.0/3.0)

    # correlation contribution is -[22*eps_c + 26*rs*(d eps_c / d rs)]/(15*n)
    if x_only:
        finf_xc = finf_x
    else:
        eps_c, d_eps_c_d_rs = LDA_derivs(dv,param=param)
        finf_c = -(22.0*eps_c + 26.0*dv.rs*d_eps_c_d_rs)/(15.0*dv.n)
        finf_xc = finf_x + finf_c

    return finf_xc

if __name__ == "__main__":


    rs_l = [1,2,3,4,5]
    g_vmc = [1.152,1.296,1.438,1.576,1.683]
    g_vmc_ucrt = [2,6,9,9,15]

    from scipy.optimize import least_squares
    import matplotlib.pyplot as plt

    rsl = np.linspace(1.,100.,5000)
    #apar, bpar, cpar = get_fxc_pars(rsl)
    apar, bpar, cpar = get_g_minus_pars(rsl,0.)
    plt.plot(rsl,(apar - cpar)/bpar)
    plt.show(); exit()

    fchi = chi_enh(rsl)

    imax = np.argmax(fchi)
    rsmax = rsl[imax]
    hmax = fchi[imax]/2.
    find_right = False
    for irs in range(rsl.shape[0]):
        tdiff = fchi[irs] - hmax
        if tdiff > 0. and (not find_right):
            ileft = irs
            find_right = True
        elif tdiff < 0. and find_right:
            iright = irs
            break
    hwhm = (rsl[iright] - rsl[ileft])/2.

    ffn = lambda c, x : c[0]/(1. + ((x - c[1])/c[2])**2)
    def obj(c):
        return ffn(c,rsl) - fchi
    res = least_squares(obj,[fchi[imax],rsmax,hwhm])
    print(res)

    plt.plot(rsl,fchi)
    plt.plot(rsl,ffn(res.x,rsl))
    #plt.scatter(rs_l,g_vmc)
    plt.show()
    exit()

    tstr = ''
    for irs, rs in enumerate(rs_l):
        enh = chi_enh(rs)
        pdiff = 200*abs(enh - g_vmc[irs])/(enh + g_vmc[irs])
        tstr += '{:} & {:}({:}) & {:.6f} & {:.2f} \\\\ \n'.format(rs,\
            g_vmc[irs], g_vmc_ucrt[irs], enh, pdiff)
    print(tstr)
