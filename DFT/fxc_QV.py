import numpy as np
from scipy.optimize import bisect
from scipy.interpolate import splrep, splev
from scipy.special import erf

from constants import pi
from DFT.asymptotics import get_fxc_high_freq
from DFT.common import xc_shear_modulus
from DFT.fxc_ALDA import fxc_ALDA
from utilities.integrators import nquad
from utilities.roots import bracket

"""
    Zhixin Qian and Giovanni Vignale,
    ``Dynamical exchange-correlation potentials for an electron liquid'',
    Phys. Rev. B 65, 235121 (2002).
    https://doi.org/10.1103/PhysRevB.65.235121
"""

# [Gamma(1/4)]**2
gamma_14_sq = 13.1450472065968728685447786119766533374786376953125


def fxcl_static(dv):
    fxc_alda = fxc_ALDA(dv, x_only = False, param = 'PW92')
    mu_xc = xc_shear_modulus(dv)
    return fxc_alda + 4.*mu_xc/(3.*dv.n)

def s_3_l(kf):
    """ Eq. 16 of Qian and Vignale """
    lam = (pi*kf)**(0.5) # lambda = 2 k_F/ k_TF
    s3l = 5. - (lam + 5./lam)*np.arctan(lam) - 2./lam*np.arcsin(lam/(1+lam**2)**(0.5)) \
        + 2./(lam*(2.+lam**2)**(0.5))*(pi/2. - np.arctan(1./(lam*(2.+lam**2)**(0.5))))
    return -s3l/(45*pi)

def get_qv_pars(dv):

    if hasattr(dv.rs,'__len__'):
        nrs = len(dv.rs)
        a3l = np.zeros(nrs)
        b3l = np.zeros(nrs)
        g3l = np.zeros(nrs)
        o3l = np.zeros(nrs)
        for irs,ars in enumerate(dv.rs):
            a3l[irs],b3l[irs],g3l[irs],o3l[irs] = get_qv_pars_single(density_variables(ars))
    else:
        a3l,b3l,g3l,o3l = get_qv_pars_single(dv)
    return a3l,b3l,g3l,o3l

def get_qv_pars_single(dv):

    c3l = 23./15. # just below Eq. 13

    s3l = s_3_l(dv.kF)
    """     Eq. 28   """
    a3l = 2*(2./(3*pi**2))**(1./3.)*dv.rs**2*s3l
    """     Eq. 29   """
    b3l = 16*(2**10/(3*pi**8))**(1./15.)*dv.rs*(s3l/c3l)**(4./5.)

    fxc_0 = fxcl_static(dv)
    fxc_inf = get_fxc_high_freq(dv, x_only = False, param = 'PW92')

    # approx. expression for mu_xc that interpolates metallic data, from Table 1 of QV
    # fitting code located in fit_mu_xc

    delta_fxc = fxc_0-fxc_inf

    def solve_g3l(tmp):
        """    Eq. 27    """
        o3l = 1 - 1.5*tmp
        o3l2 = o3l**2
        """    Eq. 30    """
        res = 4*(2*pi/b3l)**(0.5)*a3l/gamma_14_sq
        res += o3l*tmp*np.exp(-o3l2/tmp)/pi + 0.5*(tmp/pi)**(0.5)*(tmp + 2*o3l2)*(1 + erf(o3l/tmp**(0.5)))
        res *= 4*(pi/dv.n)**(0.5) # 2*omega_p(0)/n
        return delta_fxc + res

    poss_brack = bracket(solve_g3l,(1.e-6,3.0),nstep=500,vector=True)
    g3l = 1.e-14
    for tbrack in poss_brack:
        tg3l,success = bisect(solve_g3l,*tbrack,maxiter=200,full_output=True)
        if success.converged:
            g3l = max(tg3l,g3l)
    o3l = 1. - 1.5*g3l
    return a3l,b3l,g3l,o3l

def fxc_QV_im(omega,dv,pars=()):

    if len(pars)==0:
        a3,b3,g3,om3 = get_qv_pars(dv)
    else:
        a3,b3,g3,om3 = pars

    wt = omega/(2*dv.wp0)

    imfxc = a3/(1 + b3*wt**2)**(5/4)
    imfxc += wt**2*np.exp(-(np.abs(wt)-om3)**2/g3)
    imfxc *= -omega/dv.n

    return imfxc

def wrap_kram_kron(to,omega,dv):
    return fxc_QV_im(to,dv)/(to - omega)

def kram_kron(omega,dv):
    return nquad(wrap_kram_kron,('-inf','inf'),'global_adap',\
        {'itgr':'GK','prec':1.e-6,'npts': 9,'min_recur':4,'max_recur':1500,'n_extrap':400,'inf_cond':'fun'},
        pars_ops={'PV':[omega]},args=(omega,dv)
    )

def fxc_QV(omega,dv):
    im_fxc = fxc_QV_im(omega,dv)
    fxc_inf = get_fxc_high_freq(dv, x_only = False, param = 'PW92')
    if hasattr(omega,'__len__'):
        re_fxc = np.zeros(omega.shape[0])
        for iom, om in enumerate(omega):
            re_fxc[iom],terr = kram_kron(om,dv)
            if terr['code'] == 0:
                print(('WARNING, not converged for omega={:.4f}; last error {:.4e}').format(om,terr['error']))
    else:
        re_fxc, terr = kram_kron(omega,dv)
        if terr['code'] == 0:
            print(('WARNING, not converged for omega={:.4f}; last error {:.4e}').format(omega,terr['error']))
    return fxc_inf + re_fxc/pi + 1.j*im_fxc

class fxc_QV_spline:
    
    def __init__(self, opts):
        if 'spline_pars' not in opts:
            self.gen_spline_pars(opts['omega_min'], opts['omega_max'], opts['Nomega'], opts['DV'] )
        else:
            self.QV_spline_pars = opts['spline_pars']
        self.DV = opts['DV']
        return

    def gen_spline_pars(self, omega_min, omega_max, Nomega, dv):
        omega_l = np.linspace(omega_min, omega_max, Nomega)
        fxc_qv = fxc_QV(omega_l, dv)
        self.QV_spline_pars = splrep(omega_l,fxc_qv.real)
        self.freqs = omega_l
        self.fxc_real = fxc_qv.real
        self.fxc_imag = fxc_qv.imag
        return self.QV_spline_pars

    def eval(self,omega):
        return splev(omega.real, self.QV_spline_pars) + 1.j * fxc_QV_im(omega, self.DV)
    

if __name__ == "__main__":

    from DFT.common import HEG
    get_qv_pars_single(HEG(1.))