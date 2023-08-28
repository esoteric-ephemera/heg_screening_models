import multiprocessing as mp
import numpy as np
from os import system, path

from constants import pi
from DFT.chi import VCOUL, WCOUL
from DFT.common import HEG
from utilities.integrators import nquad

class FT_3D_SPH:

    def __init__(self):
        return

    def FT_3D_SPH_integrand(self,q,r):

        if self.fxc_param.upper() == 'BARE':
            func = VCOUL(q)
        else:
            func = WCOUL(q, 1.e-12j*self.DV.wp0, self.DV.rs, self.fxc_param, self.DF_type, fxc_opts = self.fxc_opts).real
        return q*np.sin(q*r)*func

    def _FT_3D_SPH(self,r,bkpt = 4., prec = 1.e-7):
        qc = bkpt*self.DV.kF
        twrap = lambda q : self.FT_3D_SPH_integrand(q,r)
        
        """
        val_1, log_d = nquad(twrap,(0.,'inf'),'global_adap',{'itgr':'GK','prec': prec,'npts':5})
        val_2 = 0.
        if log_d['code'] <= 0.:
            print(f'WARNING: 3D FT failed for r = {r}, last error = {log_d["error"]}')
        """
        val_1, log_d = nquad(twrap,(0.,qc),'global_adap',{'itgr':'GK','prec': prec/2.,'npts':5})
        if log_d['code'] <= 0.:
            print(f'WARNING: lower 3D FT failed for r = {r}, last error = {log_d["error"]}')
        twrap = lambda q : self.FT_3D_SPH_integrand(1./q,r)/q**2
        val_2, log_d = nquad(twrap,(1./(100.*qc), 1./qc),'global_adap',{'itgr':'GK','prec': prec/2.,'npts':5})
        if log_d['code'] <= 0.:
           print(f'WARNING: upper 3D FT failed for r = {r}, last error = {log_d["error"]}')        
        
        return (val_1 + val_2)/(2.*pi**2*r)

    def eval(self):
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

    def eval_WCOUL_r(self, rs,
        fxc_param, 
        DF_type, 
        fxc_opts = {},
        r_min = 0.1,
        r_max = 5.,
        Nr = 500,
        Nproc = 1
    ):
        
        self.Nproc = Nproc
        
        if fxc_param.upper() == 'BARE':
            save_dir = f'./real_space/'
        else:
            save_dir = f"./real_space/{fxc_param.replace(' ','-')}/rs_{rs}/"
        if not path.isdir(save_dir):
            system(f"mkdir -p {save_dir}")

        self.DV = HEG(rs)
        r_l = np.linspace(r_min, r_max, Nr)
        self.r = r_l / self.DV.kF
        self.Nr = Nr
        self.fxc_param = fxc_param
        self.DF_type = DF_type
        self.fxc_opts = fxc_opts

        wcr = self.eval()
        if fxc_param.upper() == 'BARE':
            svflnm = f'{save_dir}BARE.csv'
        else:
            svflnm = f'{save_dir}{fxc_param}_{DF_type}_rs-{rs}.csv'
        np.savetxt(svflnm,
            np.transpose((r_l,wcr)), delimiter=',', header = 'kF r, W(r)'
        )
        return r_l, wcr

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    rs = 10.
    _heg = HEG(rs)

    r_l, vcr = FT_3D_SPH().eval_WCOUL_r(rs,
        'BARE', 
        'TCTC', 
        fxc_opts = {},
        r_min = 0.1,
        r_max = 5.,
        Nr = 500,
        Nproc = 4
    )
    plt.plot(r_l,vcr)
    plt.plot(r_l,_heg.kF/r_l)

    r_l, wcr = FT_3D_SPH().eval_WCOUL_r(rs,
        'ALDA', 
        'TCTC', 
        fxc_opts = {},
        r_min = 0.1,
        r_max = 5.,
        Nr = 500,
        Nproc = 4
    )
    plt.plot(r_l, wcr)
    plt.plot()
    plt.show()
