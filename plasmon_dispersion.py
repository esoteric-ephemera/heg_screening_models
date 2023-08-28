import json
import numpy as np
from os import system, path
#from scipy.optimize import bisect

from DFT.common import HEG
from DFT.chi import inverse_DF
from DFT.fxc_QV import fxc_QV_spline

plas_dir = './plasmon_disp/'
plas_dir_for_taylor = './plasmon_disp_for_taylor/'

def sign(f):
    if f < 0.:
        sgn = -1.
    elif f == 0.:
        sgn = 0.
    elif f > 0.:
        sgn = 1.
    return sgn

"""
def bracket(func,x0, Nstep = 1000, scl = 0.05):

    a = x0
    sgn0 = sign(func(a))
    for isgn in [1.,-1.]:
        for istep in range(1,Nstep+1):
            b = (1. + isgn*istep*scl)*x0
            tsgn = sign(func(b))
            if sgn0*tsgn < 0.:
                break
            a = b

    return sorted([a,b])
"""
def bracket(func, x0, x1, Nstep = 200):
    xl = np.linspace(x0, x1, Nstep)
    sgn0 = sign(func(x0))

    a = x0
    for ix in range(1,Nstep):
        b = xl[ix]
        tsgn = sign(func(b))
        if sgn0*tsgn <= 0.:
            break
        sgn0 = tsgn
        a = b

    return sorted([a, b])
    
def bisect(func, a, b, Nstep = 1000, atol = 1.e-10):
    fa = func(a)
    fb = func(b)
    if fa*fb > 0.:
        raise SystemExit(f'Function values f({a}) = {fa}\n  and f({b}) = {fb}\n have same sign!')
    
    sa = sign(fa)
    sb = sign(fb)
    for istep in range(Nstep):
        mid = (a + b)/2.
        fmid = func(mid)
        if abs(fmid) < atol or abs(a - mid) < atol:
            return mid, {'success': True, 'fopt': fmid}
        smid = sign(fmid)
        if smid == sa:
            a = mid
            sa = smid
        else:
            b = mid
            sb = smid

    return mid, {'success': False, 'fopt': fmid}


def get_plasmon_dispersion(x_l, rs, fxc_param, _DF, 
    tol = 1.e-14, _plas_dir = plas_dir, save_output = True,
    masked = True
):

    logdir = f"{_plas_dir}/{fxc_param.replace(' ','-')}/rs_{rs}/"
    if not path.isdir(logdir) and save_output:
        system(f'mkdir -p {logdir}')


    _HEG = HEG(rs)
    q_l = _HEG.kF*x_l

    Nq = q_l.shape[0]
    wc_l = np.zeros(Nq)
    df_l = np.zeros(Nq)

    wc_m = _HEG.wp0

    fxc_opts = {}
    if fxc_param == 'QV spline':
        fxc_opts['omega_min'] = 0.1*_HEG.wp0
        fxc_opts['omega_max'] = 3.*_HEG.wp0
        fxc_opts['Nomega'] = 50
        fxc_opts['DV'] = _HEG
        fxc_opts['spline_pars'] = fxc_QV_spline(fxc_opts).QV_spline_pars

    for iq, aq in enumerate(q_l):

        def obj(omega):
            df = 1./inverse_DF(aq, omega + 1.e-12j * _HEG.wp0, rs, fxc_param, _DF, fxc_opts = fxc_opts)
            return df.real

        xa, xb = bracket(obj, wc_m, 1.2*wc_m, Nstep = 1000)
        if obj(xa)*obj(xb) > 0.:
            xa, xb = bracket(obj, 0.8*wc_m, wc_m, Nstep = 1000)
        
        if obj(xa)*obj(xb) > 0.:
            break
        
        #_bisected = bisect(obj, xa, xb, full_output=True, maxiter=1000, rtol=tol, xtol = tol)
        _bisected = bisect(obj, xa, xb)
        if not _bisected[1]['success']:
            print(f"WARNING: freq bisection failed, best value = {_bisected[1]['fopt']}")
        #if not _bisected[1].converged:
        #    raise SystemExit(f'FATAL, bisection could not locate zero:\n{_bisected[1].flag}')

        wc_l[iq] = _bisected[0]
        df_l[iq] = obj(_bisected[0])
        #print(aq/_HEG.kF,wc_l[iq])
        wc_m = _bisected[0]

    if masked:
        tmsk = wc_l > 0.
    else:
        tmsk = np.array([True for _ in range(Nq)])
    odat = np.transpose((x_l[tmsk],wc_l[tmsk]/_HEG.wp0,df_l[tmsk]))
    if save_output:
        np.savetxt(f'{logdir}/wc_{_DF}_rs_{rs}.csv',
            odat,
            delimiter=',',
            header='q/kF, wc(q)/wp(0), Re eps(wc(q))'
        )

    return odat

class find_flat_plasmon:

    def __init__(self,fxc, _DF ,
        q_min = 1.e-2, q_max = 0.1, Nq = 20,
        poly_degree = 4,  enforce_zeroth = True
    ): 
        self.fxc = fxc
        self.DF = _DF
        self.q_min = q_min
        self.q_max = q_max
        self.Nq = Nq
        self.poly_degree = poly_degree
        self.enforce_zeroth = enforce_zeroth

        self.wc = {}
        self.wc_fit_pars = {}
        self.q = np.linspace(self.q_min, self.q_max, self.Nq)

    def get_and_fit_wc(self,rs):
        tmp = get_plasmon_dispersion(self.q, rs, self.fxc, self.DF, 
            tol = 1.e-14, 
            _plas_dir = plas_dir_for_taylor, 
            save_output=False,
            masked=True
        )

        if len(tmp[:,0]) < 2:
            # sometimes the plasmon dispersion has a hard time,
            # so we simply just force bisection to pick a new value
            return -1.e20*sign(self._last_val[1])
        
        self.wc[rs] = list(tmp[:,1])
        tpoly_degree = max(2,min(self.poly_degree,len(self.wc[rs])))

        if self.enforce_zeroth:
            _poly_obj = np.polynomial.Polynomial.fit(
                tmp[:,0], (np.array(self.wc[rs]) - 1.)/tmp[:,0],
                tpoly_degree
            ).convert(domain=(-1,1))
            
            coefs = [1.] + list(_poly_obj.coef)

        else:
            _poly_obj = np.polynomial.Polynomial.fit(
                tmp[:,0],self.wc[rs],
                tpoly_degree
            ).convert(domain=(-1,1))
            coefs = list(_poly_obj.coef)

        self.wc_fit_pars[rs] = coefs.copy()
        #print(rs,coefs[2])
        self._last_val = (rs,2.*coefs[2])
        return 2.*coefs[2]
    
    def bracket_rs_crit(self,rs_min = 1., rs_max = 20., Nrs = 50):
        tsgn = sign(self.get_and_fit_wc(rs_min))
        rs_a = rs_min
        rs_l = np.linspace(rs_min,rs_max,Nrs)
        for irs in range(1,Nrs):
            rs_b = rs_l[irs]
            csgn = sign(self.get_and_fit_wc(rs_b))
            if csgn*tsgn < 0.:
                self.bracket = (rs_a, rs_b)
                return
            rs_a = rs_b

        return
    
    def find_rs_crit(self,rs_min = 1., rs_max = 20., Nrs = 50):
        self.bracket_rs_crit(rs_min = rs_min, rs_max = rs_max, Nrs = Nrs)

        _bisected = bisect(self.get_and_fit_wc,
            *self.bracket
        )
        if not _bisected[1]['success']:
            print(f"WARNING: rs bisection failed, best value = {_bisected[1]['fopt']}")

        """
        _bisected = bisect(self.get_and_fit_wc,
            *self.bracket, 
            full_output=True, 
            maxiter=1000,
            xtol = 1.e-12,
            rtol = 1.e-12
        )
        
        if not _bisected[1].converged:
            raise SystemExit(f'FATAL, bisection could not locate zero:\n{_bisected[1].flag}')
        """
        self.rsc = _bisected[0]
        self.get_and_fit_wc(self.rsc)
        if abs(self.wc_fit_pars[self.rsc][1]) > abs(2.*self.wc_fit_pars[self.rsc][2]):
            print(f'WARNING: {self.fxc}-{self.DF} critical rs may not be saddle-point:')
            print(f'         d omega_c(0, rs_c)/ d q = {self.wc_fit_pars[self.rsc][1]}')
            print(f'         d^2 omega_c(0, rs_c)/ d q^2 = {self.wc_fit_pars[self.rsc][2]}')
        return
    
    def save(self,filename=''):
        if not path.isdir(f"{plas_dir_for_taylor}/{self.fxc.replace(' ','-')}/"):
                system(f"mkdir -p {plas_dir_for_taylor}/{self.fxc.replace(' ','-')}/")
        if len(filename) == 0:
            filename = f"{plas_dir_for_taylor}/{self.fxc.replace(' ','-')}/{self.fxc.replace(' ','-')}_{self.DF}.json"

        td = (self.__dict__).copy()
        td['q'] = list(td['q'])
        with open(filename,'w+') as _logfl_:
            json.dump(td,_logfl_)

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    _ffp = find_flat_plasmon('GKI', 'TCTC',enforce_zeroth=False)
    _ffp.find_rs_crit()
    _ffp.save()
    print(_ffp.wc_fit_pars[_ffp.rsc])
    rs_l = []
    convex = []
    for ars in _ffp.wc_fit_pars:
        rs_l.append(ars)
        convex.append(2.*_ffp.wc_fit_pars[ars][2])
    plt.scatter(rs_l,convex)
    plt.ylim(-1.,1.)
    plt.show()

    exit()
    print(_ffp.rsc)
    print(_ffp.wc)
    exit()

    qi = np.linspace(_ffp.q_min,_ffp.q_max,50*_ffp.Nq)
    for rs in np.arange(1.,12,1):
        _ffp.get_and_fit_wc(rs)
        plt.scatter(_ffp.q,_ffp.wc[rs])
        plt.plot(qi,np.polynomial.Polynomial(_ffp.wc_fit_pars[rs])(qi))
    plt.show()
    exit()

    q_l = np.linspace(1.e-2,3.,2000)
    dat = get_plasmon_dispersion(q_l, 22., 'RPA', 'TCTC')
    plt.plot(dat[:,0],dat[:,1])
    plt.show()