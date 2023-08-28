import numpy as np

from constants import pi
from DFT.asymptotics import get_fxc_pars

def fxc_CDOP(q,dv):

    Q2 = (q/dv.kF)**2

    CA, CB, CC = get_fxc_pars(dv, param = 'PW92')

    g = CB/(CA - CC)
    alpha = 1.5*CA/(dv.rsh**(0.5)*CB*g)
    beta = 1.2/(CB*g)

    Gp = CC*Q2 + CB*Q2/(g + Q2) + alpha*Q2**2 *np.exp(-beta*Q2)

    return -4.*pi*Gp/q**2

if __name__ == "__main__":

    import matplotlib.pyplot as plt

    rs_l = [2,5,10]
    ql = np.linspace(1.,3.5,2000)

    for rs in rs_l:
        dv = {'rs': rs, 'rsh': rs**(0.5), 'n': 3./(4.*pi*rs**3),
            'kF': (9*pi/4.)**(1./3.)/rs }
        zl = ql*dv['kF']

        plt.plot(ql,g_corradini(zl,dv),label='$r_s={:}$'.format(rs))
    plt.xlim(1.,4.)
    plt.ylim(0.2,1.6)
    plt.legend()
    plt.show()
