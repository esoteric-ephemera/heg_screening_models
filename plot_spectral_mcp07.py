
import numpy as np
import matplotlib.pyplot as plt
from os import path, system
from scipy.optimize import bisect
from scipy.interpolate import splrep, splev

from constants import Hartree, Hartree_to_keV, pi
from DFT.chi import spectral_HEG
from DFT.common import HEG
from frequency_moments import freq_mom_single_rs
from main import sdir    

""" 
These are pretty bespoke functions, intended 
for a few instructive plots
"""

fwhm_dir = "./FWHM_data/"
if not path.isdir(fwhm_dir):
    system(f"mkdir {fwhm_dir}")

def spectral_plot():

    eps = 1.e-6

    fig, ax = plt.subplots(figsize=(6,4))
    rs = 69.

    heg = HEG(rs,d=3)

    ql = [heg.kF*x for x in (0.5, 1., 1.5, 2., 2.5, 2.8)]
    wl = heg.wp0*np.linspace(0.1,2.,2000)
    colors = plt.cm.inferno(np.linspace(0,0.8,len(ql)))
    for iq, q in enumerate(ql):
        s = spectral_HEG(q, wl + 1.e-15j*heg.wp0, rs, "MCP07", d = 3)
        wc = np.array([q**2/2. + (-1)**(i+1) * q*heg.kF for i in range(2)])
        sc = spectral_HEG(q, wc[wc > 0] + 1.e-15j*heg.wp0, rs, "MCP07", d = 3)

        w0 = wl[s.argmax()]
        smax = s.max()
        hmax = smax/2.
        freq_lower = wl < w0
        freq_upper = wl > w0

        plabel = "$q/k_\\mathrm{F}=" + f"{q/heg.kF}$"
        ax.plot(wl/heg.wp0,s/Hartree_to_keV,color=colors[iq],label=plabel)
        ax.scatter(wc[wc>0]/heg.wp0,sc/Hartree_to_keV,color=colors[iq],marker="*",s=40) 

        if len(wl[freq_lower]) > 0 and len(wl[freq_upper]) > 0:
            hmindx = [
                np.argmin(np.abs(s[freq_lower] - hmax)),
                np.argmin(np.abs(s[freq_upper] - hmax))
            ]

            w_hmax = np.array([wl[freq_lower][hmindx[0]], wl[freq_upper][hmindx[1]]])
                    
            for i in range(2):
                
                if i == 0:
                    lbd = 0.8*w_hmax[i]
                    ubd = w0
                elif i == 1:
                    lbd = w0
                    ubd = 1.1*w_hmax[i]

                w_hmax[i] = bisect(
                    lambda x : splev(x,splrep(wl,s)) - hmax,
                    lbd, ubd
                )

            s_hmax = spectral_HEG(q, w_hmax + 1.e-15j*heg.wp0, rs, "MCP07", d = 3)
            ax.scatter(w_hmax/heg.wp0,s_hmax/Hartree_to_keV,color=colors[iq],marker="s",s=40) 


        label_pos_d = {
            0.5: (0.4,2.5),
            1.0: (.35,35.5),
            1.5 : (.35,2425), 
            2.0 : (1.03,1070.), 
            2.5 : (0.16,3.8e5),
            2.8 : (.23,468.)
        }
        for x in label_pos_d:
            if abs(q/heg.kF - x) < eps:
                label_pos = label_pos_d[x]
                break

        ax.annotate(
            plabel,
            label_pos,
            color=colors[iq],
            fontsize=12
        )

        #fwhm = w_hmax[1] - w_hmax[0]
        #sigma = fwhm/(2.*(2*np.log(2))**(0.5))


    #ax.legend(fontsize=12)

    ax.set_xlabel(r"$\omega/\omega_p(0)$", fontsize=12)
    ax.set_xlim(wl.min()/heg.wp0,wl.max()/heg.wp0)

    ax.set_ylabel("$S_\\mathrm{MCP07}(q,\\omega)$ (1/keV)", fontsize=12)
    ax.set_ylim(1.,1.e6)   
    ax.set_yscale("log")

    #plt.show();exit()
    plt.savefig(f"{sdir}/specrtral_MCP07_rs_69.pdf",dpi=600,bbox_inches="tight")

def fwhm(
    q_min : float = 0.1,
    q_max : float  = 3.,
    Nq : int = 500,
    fxc : str = "MCP07",
    plot_moments : bool = False
):

    fig, ax = plt.subplots(figsize=(6,4))
    

    rsl = [ 22, 69]
    colors = plt.cm.inferno(np.linspace(0,0.8,len(rsl)))
    for irs, rs in enumerate(rsl):

        heg = HEG(rs,d=3)

        ql = heg.kF*np.linspace(q_min, q_max, Nq)
        wl = heg.wp0*np.linspace(0.1,2.,2000)
        fwhm = np.zeros(ql.shape)

        if plot_moments:
            moms = []
            for imom in range(3):

                _, tspec = freq_mom_single_rs(q_min, q_max, Nq, rs, imom, fxc, fxc_opts = {})
                moms.append(tspec)
            
            avg_w = moms[1]/(moms[0])
            dev_w = np.maximum(0., moms[2]/moms[0] - avg_w**2)**(0.5)
            
            label = None
            if irs == 0:
                label = r"$\langle \Delta \omega (q) \rangle$"
            ax.plot(ql/heg.kF,dev_w/heg.wp0,color=colors[irs],linestyle=":",label = label)

        for iq, q in enumerate(ql):

            s = spectral_HEG(q, wl + 1.e-15j*heg.wp0, rs, fxc, d = 3)
            #wc = np.array([q**2/2. + (-1)**(i+1) * q*heg.kF for i in range(2)])

            w0 = wl[s.argmax()]
            smax = s.max()
            hmax = smax/2.
            freq_lower = wl < w0
            freq_upper = wl > w0

            if len(wl[freq_lower]) > 0 and len(wl[freq_upper]) > 0:
                hmindx = [
                    np.argmin(np.abs(s[freq_lower] - hmax)),
                    np.argmin(np.abs(s[freq_upper] - hmax))
                ]

                w_hmax = np.array([wl[freq_lower][hmindx[0]], wl[freq_upper][hmindx[1]]])
                        
                for i in range(2):
                    
                    if i == 0:
                        lbd = 0.8*w_hmax[i]
                        ubd = w0
                    elif i == 1:
                        lbd = w0
                        ubd = 1.1*w_hmax[i]

                    hwhm = lambda x : splev(x,splrep(wl,s)) - hmax

                    if hwhm(lbd)*hwhm(ubd) < 0.:
                        w_hmax[i] = bisect(
                            hwhm, lbd, ubd
                        )

                fwhm[iq] = w_hmax[1] - w_hmax[0]
        
        ax.plot(ql/heg.kF,fwhm/heg.wp0,color=colors[irs],label = "$r_\\mathrm{s}=" + f"{rs}$")
        np.savetxt(
            f"{fwhm_dir}/FWHM-rs_{rs}-fxc_{fxc}.csv",
            np.transpose((ql/heg.kF,fwhm/heg.wp0)),
            delimiter=",",
            header="q/kF, w_life / wp0"
        )

    ax.legend(title=fxc, title_fontsize=12, fontsize=12, frameon=False)

    ax.set_xlabel(r"$q/k_\mathrm{F}$", fontsize=12)
    ax.set_xlim(q_min,q_max)

    ax.set_ylabel("Lifetime [$\\omega_p(0)$]", fontsize=12)
    ax.set_ylim(0.,1.75)

    #plt.show();exit()
    plt.savefig(f"{sdir}/lifetime_MCP07.pdf",dpi=600,bbox_inches="tight")


if __name__ == "__main__":

    #spectral_plot()
    fwhm(plot_moments=True)