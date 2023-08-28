
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

    _heg = HEG(rs)
    ql = _heg.kF*np.linspace(q_min,q_max,Nq)
    moment = np.zeros(Nq)
    for iq, aq in enumerate(ql):

        moment[iq] = freq_mom_single_q_single_rs(aq, rs, MOM, fxc_param, fxc_opts = fxc_opts)#, min_dens = wdens, min_freq_ubd=freq_max)

    np.savetxt(sfl,np.transpose((ql/_heg.kF,moment)),
        delimiter=',',
        header=f'q/kF, int w**{MOM} * S(q; w) dw'
    )

    return ql/_heg.kF,moment

def freq_mom_single_q_single_rs(q, rs, MOM, fxc_param, 
    fxc_opts = {}, tol = 5.e-7,
    alpha_m4 = .5, zeta_m4 = 0.5,
):
    _HEG = HEG(rs)
    igrd = lambda w: w**MOM * spectral_HEG(q,w + 1.e-12j*_HEG.wp0, rs, fxc_param, fxc_opts = fxc_opts)

    out_d = {}
    wl, dw = m4_grid(maxN, zeta=zeta_m4*_HEG.wp0, alpha = alpha_m4*_HEG.wp0)

    return np.sum(dw*igrd(wl))


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    rs = 69
    qls = []
    moms = []

    _HEG_ = HEG(rs)
    for iMOM in range(3):
        ql, tspec = freq_mom_single_rs(0.01, 4., 400, rs, iMOM, 'MCP07')
        moms.append(tspec)

    avg_w = moms[1]/(moms[0]*_HEG_.wp0)
    dev_w = np.maximum(0., moms[2]/(moms[0]*_HEG_.wp0**2) - avg_w**2)**(0.5)
    plt.plot(ql,avg_w)
    plt.plot(ql,dev_w)

    plt.plot(ql,(ql**2/2. + _HEG_.kF*ql)*_HEG_.kF**2/_HEG_.wp0)

    if path.isfile(f'.for_freq_mom_test/MCP07_Sq_rs_{rs}_original.csv'):
        q, og_s = np.transpose(np.genfromtxt(f'.for_freq_mom_test/MCP07_Sq_rs_{rs}_original.csv',delimiter=',',skip_header=1))
        _, m1 = np.transpose(np.genfromtxt(f'.for_freq_mom_test/MCP07_moment_1.0_rs_{rs}_original.csv',delimiter=',',skip_header=1))
        _, m2 = np.transpose(np.genfromtxt(f'.for_freq_mom_test/MCP07_moment_2.0_rs_{rs}_original.csv',delimiter=',',skip_header=1))

        plt.plot(q,m1/og_s,color='blue',linestyle=':')
        plt.plot(q,np.maximum(0., m2/og_s  - (m1/og_s)**2)**(0.5),color='orange',linestyle=':')

    plt.ylim(0.,5.)

    plt.show()