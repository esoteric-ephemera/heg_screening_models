from constants import pi, c_kF_rs

# NB: got GAMMA_GKI value from julia using the following script
# using SpecialFunctions
# BigFloat((gamma(0.25))^2/(32*pi)^(0.5))
GAMMA_GKI = 1.311028777146059809410871821455657482147216796875

C_GKI = 23.*pi/15.

class HEG:

    def __init__(self, rs: float, d : int = 3):
        self.rs = rs
        self.rsh = rs**(0.5)

        if d == 2:
            self._HEG_2D()
        elif d == 3:
            self._HEG_3D()

    def _HEG_3D(self):
        self.n = 3./(4.*pi*self.rs**3)
        self.kF = c_kF_rs/self.rs
        self.ks = (4.*self.kF/pi)**(0.5)
        self.epsF = self.kF**2/2.
        self.wp0 = (4.*pi*self.n)**(0.5)

    def _HEG_2D(self):
        self.n = 1./(pi*self.rs**2)
        self.kF = 2**(0.5)/self.rs
        self.ks = 2.
        self.epsF = self.kF**2/2.
        self.wp0 = 0.

    def as_dict(self):
        return {'rs': self.rs, 'rsh': self.rsh, 'n': self.n, 'kF': self.kF, 
            'epsF': self.epsF, 'omega_p(0)': self.wp0
        }
    
def xc_shear_modulus(dv,pars = [0.031152,0.011985,2.267455]):
    mu_xc_per_n = pars[0]/dv.rs + (pars[1] - pars[0])*dv.rs/(pars[2] + dv.rs**2)
    return mu_xc_per_n*dv.n