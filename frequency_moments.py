
from itertools import product
import numpy as np
from os import path, system

from DFT.chi import spectral_HEG
from DFT.common import HEG
from utilities.gauss_cheb_quad import m4_grid, maxN

odir = './frequency_moments/'

def freq_mom_single_rs(q_min, q_max, Nq, rs, MOM, fxc_param, fxc_opts= {}):

    sdir = f'{odir}/{fxc_param}/rs_{rs}/'
    sfl = f'{sdir}moment_{MOM}.csv'
    system(f'mkdir -p {sdir}')

    _heg = HEG(rs,d=3)
    ql = _heg.kF*np.linspace(q_min,q_max,Nq)
    moment = np.zeros(Nq)
    for iq, aq in enumerate(ql):

        moment[iq] = freq_mom_single_q_single_rs(aq, rs, MOM, fxc_param, fxc_opts = fxc_opts)

    np.savetxt(sfl,np.transpose((ql/_heg.kF,moment)),
        delimiter=',',
        header=f'q/kF, int w**{MOM} * S(q; w) dw'
    )

    return ql/_heg.kF,moment

def freq_mom_single_q_single_rs(q, rs, MOM, fxc_param, 
    fxc_opts = {}, b1 = 1.e-1, b2 = 1.e-3,
    alpha_m4 = .5, zeta_m4 = 0.5,
):
    _HEG = HEG(rs,d=3)
    def igrd(w,broaden):
        return w**MOM * spectral_HEG(q,w + 1.j*broaden*_HEG.wp0, rs, fxc_param, fxc_opts = fxc_opts)
    wl, dw = m4_grid(maxN, zeta=zeta_m4*_HEG.wp0, alpha = alpha_m4*_HEG.wp0)

    i1 = np.sum(dw*igrd(wl,b1))
    i2 = np.sum(dw*igrd(wl,b2))
    # extrapolate to zero imaginary frequency
    return i2 - b2*(i2 - i1)/(b2 - b1)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    rs = 4
    fxc = "MCP07"
    qls = []
    moms = []

    _HEG_ = HEG(rs,d=3)
    for iMOM in range(3):
        ql, tspec = freq_mom_single_rs(0.01, 4., 400, rs, iMOM, fxc)
        moms.append(tspec)

    avg_w = moms[1]/(moms[0]*_HEG_.wp0)
    dev_w = np.maximum(0., moms[2]/(moms[0]*_HEG_.wp0**2) - avg_w**2)**(0.5)
    plt.plot(ql,avg_w,color="darkblue")
    plt.plot(ql,dev_w,color="darkorange")

    plt.plot(ql,(ql**2/2. + _HEG_.kF*ql)*_HEG_.kF**2/_HEG_.wp0,color="green")

    if path.isfile(f'./freq_moment_test/{fxc}_Sq_rs_{rs}_original.csv'):
        q, og_s = np.transpose(np.genfromtxt(f'./freq_moment_test/{fxc}_Sq_rs_{rs}_original.csv',delimiter=',',skip_header=1))
        _, m1 = np.transpose(np.genfromtxt(f'./freq_moment_test/{fxc}_moment_1.0_rs_{rs}_original.csv',delimiter=',',skip_header=1))
        _, m2 = np.transpose(np.genfromtxt(f'./freq_moment_test/{fxc}_moment_2.0_rs_{rs}_original.csv',delimiter=',',skip_header=1))

        plt.plot(q,m1/og_s,color='cyan',linestyle=':')
        plt.plot(q,np.maximum(0., m2/og_s  - (m1/og_s)**2)**(0.5),color='yellow',linestyle=':')

    plt.ylim(0.,5.)

    plt.show()