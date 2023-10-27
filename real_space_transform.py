import multiprocessing as mp
import numpy as np
from os import system, path
from scipy.fft import fht, fhtoffset

from constants import pi
from DFT.chi import VCOUL, WCOUL
from DFT.common import HEG
from utilities.integrators import nquad

class FT_SPH:

    def __init__(self,
            rs,
            fxc_param, 
            DF_type,
            d = 3,
            fxc_opts = {}
        ):
        self.rs = rs
        self.DV = HEG(rs,d=2)
        self.fxc_param = fxc_param
        self.DF_type = DF_type
        self.fxc_opts = fxc_opts
        self.d = d

    def _FT_3D_SPH_integrand(self,q,r):

        if self.fxc_param.upper() == 'BARE':
            func = VCOUL(q,d=3)
        else:
            func = WCOUL(q, 1.e-12j*self.DV.wp0, self.rs, 
                self.fxc_param, self.DF_type, 
                fxc_opts = self.fxc_opts, d=3
            ).real
        return q*np.sin(q*r)*func

    def _FT_3D_SPH(self,r,bkpt = 4., prec = 1.e-7):
        qc = bkpt*self.DV.kF
        twrap = lambda q : self._FT_3D_SPH_integrand(q,r)
        
        """
        val_1, log_d = nquad(twrap,(0.,'inf'),'global_adap',{'itgr':'GK','prec': prec,'npts':5})
        val_2 = 0.
        if log_d['code'] <= 0.:
            print(f'WARNING: 3D FT failed for r = {r}, last error = {log_d["error"]}')
        """
        val_1, log_d = nquad(twrap,(0.,qc),'global_adap',{'itgr':'GK','prec': prec/2.,'npts':5})
        if log_d['code'] <= 0.:
            print(f'WARNING: lower 3D FT failed for r = {r}, last error = {log_d["error"]}')
        twrap = lambda q : self._FT_3D_SPH_integrand(1./q,r)/q**2
        val_2, log_d = nquad(twrap,(1./(100.*qc), 1./qc),'global_adap',{'itgr':'GK','prec': prec/2.,'npts':5})
        if log_d['code'] <= 0.:
           print(f'WARNING: upper 3D FT failed for r = {r}, last error = {log_d["error"]}')        
        
        return (val_1 + val_2)/(2.*pi**2*r)

    def eval(self):

        if self.d == 2:
            
            q = np.logspace(-7,4,self.Nr)/self.rs

            if self.fxc_param.upper() == 'BARE':
                func = VCOUL(q,d=2)
            else:
                func = WCOUL(q, 1.e-12j*self.DV.wp0, self.rs, self.fxc_param, 
                    self.DF_type, fxc_opts = self.fxc_opts, d=2).real

            log_step = np.log(q[1]/q[0])
            offset = fhtoffset(log_step, mu=0., initial=-6*np.log(10))
            self.r = np.exp(offset)/q[::-1]
            Aq = q*func
            Fr = fht(Aq, log_step, mu = 0., offset = offset)/(2.*pi*self.r)

        else:

            if self.Nr > 1:
                if self.Nproc > 1:
                    with mp.Pool(min(self.Nr,self.Nproc)) as wkrs:
                        Fr = np.array(wkrs.map(self._FT_3D_SPH,self.r))
                else:
                    Fr = np.zeros(self.Nr)
                    for ir in range(self.Nr):
                        Fr[ir] = self._FT_3D_SPH(self.r[ir])
            else:
                Fr = self._FT_3D_SPH(self.r)
        return Fr

    def eval_WCOUL_r(self,
        r_min = 0.1,
        r_max = 5.,
        Nr = 500,
        Nproc = 1
    ):
        
        self.Nproc = Nproc
        
        base_dir = "./real_space"
        if self.d == 2:
            base_dir += "_2D"
        base_dir += "/"

        if self.fxc_param.upper() == 'BARE':
            save_dir = base_dir
        else:
            save_dir = f"{base_dir}{self.fxc_param.replace(' ','-')}/rs_{self.rs}/"
        if not path.isdir(save_dir):
            system(f"mkdir -p {save_dir}")

        self.DV = HEG(self.rs,d=self.d)
        if self.d == 3:
            r_l = np.linspace(r_min, r_max, Nr)
            self.r = r_l / self.DV.kF

        self.Nr = Nr

        wcr = self.eval()
        if self.d == 2:
            r_l = self.r*self.DV.kF

        if self.fxc_param.upper() == 'BARE':
            svflnm = f'{save_dir}BARE.csv'
        else:
            svflnm = f'{save_dir}{self.fxc_param}_{self.DF_type}_rs-{self.rs}.csv'
        np.savetxt(svflnm,
            np.transpose((r_l,wcr)), delimiter=',', header = 'kF r, W(r)'
        )
        return r_l, wcr

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    """
    rs = 10.
    _heg = HEG(rs,d=3)

    r_l, vcr = FT_SPH(
        rs,
        'BARE', 
        'TCTC',
        d = 3,
        fxc_opts = {},    
    ).eval_WCOUL_r(
        r_min = 0.1,
        r_max = 5.,
        Nr = 500,
        Nproc = 4
    )
    plt.plot(r_l,vcr)
    plt.plot(r_l,_heg.kF/r_l)

    r_l, wcr = FT_SPH(
        rs,
        'ALDA', 
        'TCTC', 
        d = 3
    ).eval_WCOUL_r(
        fxc_opts = {},
        r_min = 0.1,
        r_max = 5.,
        Nr = 500,
        Nproc = 4
    )
    plt.plot(r_l, wcr)
    plt.plot()
    plt.show()
    """

    rs = 4.
    _heg = HEG(rs,d=2)

    r_l, vcr = FT_SPH(
        rs,
        'BARE', 
        'TCTC',
        d = 2,
        fxc_opts = {}
    ).eval_WCOUL_r(
        r_min = 0.1,
        r_max = 5.,
        Nr = 5000,
        Nproc = 4
    )
    plt.plot(r_l,vcr)
    plt.plot(r_l,_heg.kF/r_l,linestyle=":")

    r_l, wcr = FT_SPH(
        rs,
        'ALDA', 
        'TCTC', 
        d = 2,
        fxc_opts = {}
    ).eval_WCOUL_r(
        r_min = 0.1,
        r_max = 5.,
        Nr = 5000,
        Nproc = 4
    )
    plt.plot(r_l, wcr)
    plt.ylim(-1,5)
    plt.plot()
    plt.show()
