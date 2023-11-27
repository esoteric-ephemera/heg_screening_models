
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np

from DFT.common import HEG
from frequency_moments import freq_mom_single_rs
from plot_spectral_mcp07 import fwhm_dir

font = {'family': 'serif', 'serif': ['Palatino']}
plt.rc('font', **font)
plt.rc('text', usetex=True)


def graphical_abstract(
    rs : float = 69, 
    fxc_param : str = "MCP07", 
    fxc_opt : dict = {},
    q_min : float = 0.05, q_max : float = 4., Nq : int = 500
):

    _HEG_ = HEG(rs)

    fig, ax = plt.subplots(figsize = (8.0139, 6.2739))

    colors = [("darkblue","tab:blue"),("darkorange","tab:orange")]
    moms = []
    for imom in range(3):

        ql, tspec = freq_mom_single_rs(q_min, q_max, Nq, rs, imom, fxc_param, fxc_opts = fxc_opt)
        moms.append(tspec)
    
    avg_w = moms[1]/(moms[0]*_HEG_.wp0)
    dev_w = np.maximum(0., moms[2]/(moms[0]*_HEG_.wp0**2) - avg_w**2)**(0.5)

    ax.plot(ql,avg_w,color="darkblue",label=fxc_param + r" $\langle \omega_p(q) \rangle$")
    ax.fill_between(ql, avg_w-dev_w, avg_w+dev_w,color="tab:blue", alpha = 0.25)

    fwhm_data = np.genfromtxt(
        f"{fwhm_dir}/FWHM-rs_{rs}-fxc_{fxc_param}.csv",
        delimiter=",",
        skip_header=1
    )

    ax.plot(fwhm_data[:,0],fwhm_data[:,1],color="darkorange",linestyle="--",label="Lifetime")

    cutoff = [(ql**2/2. + (-1)**i * ql)*_HEG_.kF**2/_HEG_.wp0 for i in range(2)]
    ax.plot(ql,cutoff[0], color='tab:green', label='$q^2/2 \\pm k_\\mathrm{F} q$')
    ax.plot(ql,cutoff[1], color='tab:green', linestyle="-")
    
    ax.set_xlim(q_min,q_max)
    ax.set_xlabel('$q/k_\\mathrm{F}$',fontsize=20)

    ax.set_ylabel(r'Excitation frequency $[\omega_\mathrm{p}(0)]$',fontsize=20)
    ax.set_ylim(0., 1.02*(avg_w+dev_w).max())
    
    ax.legend(fontsize=20,
        title='$r_\\mathrm{s}=' + f'{rs}$ HEG', title_fontsize=20,
        frameon = False, loc = 'upper left'
    )

    tmp_x_tcks = ax.get_xticks()
    dtck = tmp_x_tcks[1] - tmp_x_tcks[0]
    ax.xaxis.set_minor_locator(MultipleLocator(dtck/2.))
    ax.xaxis.set_major_locator(MultipleLocator(dtck))

    tmp_y_tcks = ax.get_yticks()
    dtck = tmp_y_tcks[1] - tmp_y_tcks[0]
    ax.yaxis.set_minor_locator(MultipleLocator(dtck/2.))
    ax.yaxis.set_major_locator(MultipleLocator(dtck))

    ax.annotate("Plasmons and quasi-excitons",(1.8,1.),rotation=50,fontsize=20)
    ax.annotate("Particle-hole continuum",(2.24,1.1),rotation=50,fontsize=20)
    ax.annotate("Static CDW",(2.23,0.02),xytext=(3.0,0.24),
        rotation=0, fontsize=20, arrowprops={"shrink": .05, "facecolor": "k"}
    )

    ax.tick_params(which="both",labelsize=18)


    #plt.show() ; exit()
    plt.savefig("./graphical_abstract.jpeg",dpi=600,bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close()

if __name__ == "__main__":
    graphical_abstract()