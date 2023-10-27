
import numpy as np
from os import path, system
from scipy.optimize import newton

from DFT.chi import TCTE_DF
from DFT.common import HEG

plasmon_dir = "./plasmon_dispersion/"

class plasmon_disperson:
    
    def __init__(
        self,
        fxc_param : str,
        fxc_opts : dict = {},
        d : int = 3,
    ):
        self.fxc_param = fxc_param
        self.fxc_opts = fxc_opts
        self.d = d
        
        self.save_dir = f"{plasmon_dir}/{fxc_param}/"
        if not path.isdir(self.save_dir):
            system(f"mkdir -p {self.save_dir}")


    def objective(self, freq_vec, q, rs):
        DF = TCTE_DF(q,
            freq_vec[0] + 1.j*freq_vec[1],
            rs,
            self.fxc_param,
            fxc_opts = self.fxc_opts, 
            d = self.d
        )
        return np.array([DF.real, DF.imag])
        
    def compute_plasmon_dispersion(
        self,
        qmin: float, 
        qmax: float, 
        Nq: int, 
        rs: float
    ):
        heg = HEG(rs,d=self.d)
        ql = heg.kF*np.linspace(qmin,qmax,Nq)
        wpch = ql**2/2. + ql*heg.kF

        if self.d == 2:
            wp_scale = heg.epsF
            header_str = "q/kF, wp(q)/epsF"
            wp0 = [0.01,-0.01]
        elif self.d == 3:
            wp_scale = heg.wp0
            header_str = "q/kF, wp(q)/wp(0)"
            wp0 = [heg.wp0, -0.1*heg.wp0]            

        wp = np.zeros(Nq,dtype=complex)
        for iq in range(Nq):
            wp0_new = newton(
                self.objective,
                wp0,
                args = (ql[iq],rs,),
                maxiter=1000
            )

            delta = sum(abs(wp0_new[i] - wp0[i]) for i in range(2))
            if (np.abs(wp0_new[0] + 1.j*wp0_new[1]) <= wpch[iq]) or (delta > 1.e2*wp_scale):
                break
            wp[iq] = wp0_new[0] + 1.j*wp0_new[1]
            wp0 = wp0_new.copy()
            
        q = ql[:iq]/heg.kF
        wpq = wp[:iq]/wp_scale
        np.savetxt(
            f"{self.save_dir}/wp_rs-{rs}.csv",
            np.transpose((q, wpq)),
            delimiter=",",
            header=header_str
        )
        return q, wpq

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    PDISP = plasmon_disperson("AKCK",d=3)
    q, wp = PDISP.compute_plasmon_dispersion(.1,3.,500,30.)

    plt.plot(q,wp.real)
    plt.plot(q,wp.imag)
    plt.show()