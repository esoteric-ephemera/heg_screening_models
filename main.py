import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from os import system, path

from DFT.chi import WCOUL, VCOUL, inverse_DF
from DFT.common import HEG
from frequency_moments import freq_mom_single_rs, odir
from plasmon_dispersion import plas_dir, plas_dir_for_taylor, get_plasmon_dispersion, find_flat_plasmon
from real_space_transform import FT_3D_SPH

font = {'family': 'serif', 'serif': ['Palatino']}
plt.rc('font', **font)
plt.rc('text', usetex=True)

sdir = './figs/'
if not path.isdir(sdir):
    system(f'mkdir {sdir}')

atol = 1.e-6

all_fxc = ['RPA','ALDA','CDOP','Static MCP07', 'AKCK', 'GKI', 'QV', 'QV spline', 'MCP07', 'rMCP07']

style_d = {
    'RPA': ('k','-'),
    'ALDA': ('k', ':'),
    'CDOP': ('darkblue','-'),
    'Static MCP07': ('goldenrod','--'),
    'AKCK': ('dodgerblue','-.'),
    'GKI': ('forestgreen','-'),
    'ALDA-shear': ('darkgreen','--'),
    'QV': ('darkgreen','--'),
    'QV spline': ('darkgreen','-.'),
    'MCP07': ('goldenrod','-'),
    'rMCP07': ('goldenrod',':')
}
"""
clist = plt.cm.viridis(np.linspace(0.,1.,len(all_fxc)))
lsls = ['-',':','--','-.']
style_d = {}
for ifxc, fxc in enumerate(all_fxc):
    style_d[fxc] = (clist[ifxc],lsls[ifxc%len(lsls)])
"""
def W_plots(rs, fxc_l,
    q_min = 1.e-3, q_max = 5., Nq = 5000,
):

    DV = HEG(rs)
    x_l = np.linspace(q_min, q_max, Nq)
    q_l = DV.kF*x_l
    
    wscl = 4*np.pi/DV.ks**2

    fig, ax = plt.subplots(2,1,figsize=(4,6))

    DF_l = ['TCTC','TCTE']

    for iax in range(2):
        ax[iax].plot(x_l,-VCOUL(q_l)/wscl, color = 'darkorange', linestyle = '-', label = 'Bare Coulomb')

    for fxc in fxc_l:

        for iax, wvar in enumerate(DF_l):
            ax[iax].plot(x_l, -WCOUL(q_l, 1.e-12j, rs, fxc, wvar).real/wscl, 
                color=style_d[fxc][0], linestyle=style_d[fxc][1], 
                label=fxc
            )

    if abs(rs - 22.) < atol:
        ncol = 2
        lloc = 'upper center'
        bds = [(-1200./wscl, 1200./wscl) for _ in range(2)]
    elif abs(rs - 4.) < atol:
        ncol = 1
        lloc = 'lower right'
        bds = [(-1.2,0.) for _ in range(2)]
    else:
        ncol = 2
        lloc = 'upper center'
        bds = [ax[iax].get_ylim() for iax in range(2)]
    
    for iax, wvar in enumerate(DF_l):
        ax[iax].set_xlim(q_min,q_max)
        ax[iax].hlines(0.,q_min, q_max, color = 'gray', linestyle='-', linewidth=1)

        ax[iax].set_ylim(*bds[iax])
        #ax[iax].set_ylabel('$-W_\\mathrm{' + f'{wvar}'+'}(q,\omega=0,r_\\mathrm{s})$',fontsize=14)
        ax[iax].set_ylabel(r'$-k_\mathrm{s}^2 \, W_\mathrm{' + f'{wvar}'+r'}(q)/(4\pi)$',fontsize=14)

        tmp_x_tcks = ax[iax].get_xticks()
        dtck = tmp_x_tcks[1] - tmp_x_tcks[0]
        ax[iax].xaxis.set_minor_locator(MultipleLocator(dtck/2.))
        ax[iax].xaxis.set_major_locator(MultipleLocator(dtck))

        tmp_y_tcks = ax[iax].get_yticks()
        dtck = tmp_y_tcks[1] - tmp_y_tcks[0]
        ax[iax].yaxis.set_minor_locator(MultipleLocator(dtck/2.))
        ax[iax].yaxis.set_major_locator(MultipleLocator(dtck))

    ax[1].set_xlabel('$q/k_\\mathrm{F}$',fontsize=14)
    ax[1].legend(title = '$r_\\mathrm{s}='+f'{rs}$',title_fontsize=10,fontsize=10, 
        ncol = ncol, frameon= False,
        loc = lloc#, bbox_to_anchor = (0.0,1.01)
    )
    
    ax[0].annotate('(a)',(-.2,1.),xycoords = 'axes fraction', annotation_clip = False,fontsize=20)
    ax[1].annotate('(b)',(-.2,1.),xycoords = 'axes fraction', annotation_clip = False,fontsize=20)

    #plt.show() ; exit()

    plt.savefig(f'{sdir}/WCOUL-rs_{rs}.pdf',dpi=600,bbox_inches='tight')
    
    plt.cla()
    plt.clf()
    plt.close()

    return

def DF_plots(rs, fxc_l,
    q_min = 1.e-3, q_max = 5., Nq = 5000,
):

    DV = HEG(rs)
    x_l = np.linspace(q_min, q_max, Nq)
    q_l = DV.kF*x_l

    fig, ax = plt.subplots(2,1,figsize=(4,6))

    DF_l = ['TCTC','TCTE']

    for fxc in fxc_l:

        for iax, wvar in enumerate(DF_l):

            ax[iax].plot(x_l, inverse_DF(q_l, 1.e-12j, rs, fxc, wvar).real, 
                color=style_d[fxc][0], linestyle=style_d[fxc][1], 
                label=fxc
            )
    
    for iax, wvar in enumerate(DF_l):
        ax[iax].set_xlim(q_min,q_max)
        ax[iax].hlines(0.,q_min, q_max, color = 'gray', linestyle='-', linewidth=1)

        if wvar == 'TCTC':
            ylab = '$\\epsilon^{-1}(q,\omega=0,r_\\mathrm{s})$'
        elif wvar == 'TCTE':
            ylab = r'$\widetilde\epsilon^{-1}(q,\omega=0,r_\mathrm{s})$'

        ax[iax].set_ylabel(ylab,fontsize=14)

        tmp_x_tcks = ax[iax].get_xticks()
        dtck = tmp_x_tcks[1] - tmp_x_tcks[0]
        ax[iax].xaxis.set_minor_locator(MultipleLocator(dtck/2.))
        ax[iax].xaxis.set_major_locator(MultipleLocator(dtck))

        tmp_y_tcks = ax[iax].get_yticks()
        dtck = tmp_y_tcks[1] - tmp_y_tcks[0]
        ax[iax].yaxis.set_minor_locator(MultipleLocator(dtck/2.))
        ax[iax].yaxis.set_major_locator(MultipleLocator(dtck))

    if abs(rs - 22.) < atol:
        ncol = 2
        bds = [(-2.02,1.02),(-.52,1.02)]
    elif abs(rs - 4.) < atol:
        ncol = 1
        bds = [(0.,1.02),(0.,1.15)]
    else:
        ncol = 2
        bds = [ax[i].get_ylim() for i in range(2)]

    for iax in range(2):
        ax[iax].set_ylim(*bds[iax])

    ax[1].set_xlabel('$q/k_\\mathrm{F}$',fontsize=14)
    ax[1].legend(title = '$r_\\mathrm{s}='+f'{rs}$',title_fontsize=10,fontsize=10, 
        ncol = ncol, loc = 'lower right', 
        frameon=False
    )
    
    ax[0].annotate('(a)',(-.2,.9),xycoords = 'axes fraction', annotation_clip = False,fontsize=20)
    ax[1].annotate('(b)',(-.2,.9),xycoords = 'axes fraction', annotation_clip = False,fontsize=20)

    #plt.show() ; exit()
    
    plt.savefig(f'{sdir}/DF-rs_{rs}.pdf',dpi=600,bbox_inches='tight')
    
    plt.cla()
    plt.clf()
    plt.close()
 
    return

def DF_pair_plot(rs1,rs2, fxc_l,
    q_min = 1.e-3, q_max = 5., Nq = 5000,
):
    
    fig, ax = plt.subplots(2,2,figsize=(8,6))
    lbl_from_idx = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}

    x_l = np.linspace(q_min, q_max, Nq)
    DF_l = ['TCTC','TCTE']

    for irs, rs in enumerate([rs1,rs2]):

        DV = HEG(rs)
        
        q_l = DV.kF*x_l

        for fxc in fxc_l:

            for iax, wvar in enumerate(DF_l):

                ax[iax, irs].plot(x_l, DV.ks**2*inverse_DF(q_l, 1.e-12j, rs, fxc, wvar).real, 
                    color=style_d[fxc][0], linestyle=style_d[fxc][1], 
                    label=fxc
                )
        
        for iax, wvar in enumerate(DF_l):
            ax[iax, irs].set_xlim(q_min,q_max)
            ax[iax, irs].hlines(0.,q_min, q_max, color = 'gray', linestyle='-', linewidth=1)

            if wvar == 'TCTC':
                ylab = r'$k_\mathrm{s}^2 \, \epsilon^{-1}(q)$'#,\omega=0,r_\mathrm{s})$'
            elif wvar == 'TCTE':
                ylab = r'$k_\mathrm{s}^2 \, \widetilde\epsilon^{-1}(q)$'#,\omega=0,r_\mathrm{s})$'

            ax[iax, irs].set_ylabel(ylab,fontsize=14)

            tmp_x_tcks = ax[iax, irs].get_xticks()
            dtck = tmp_x_tcks[1] - tmp_x_tcks[0]
            ax[iax, irs].xaxis.set_minor_locator(MultipleLocator(dtck/2.))
            ax[iax, irs].xaxis.set_major_locator(MultipleLocator(dtck))

            tmp_y_tcks = ax[iax, irs].get_yticks()
            dtck = .25#tmp_y_tcks[1] - tmp_y_tcks[0]
            ax[iax, irs].yaxis.set_minor_locator(MultipleLocator(dtck/2.))
            ax[iax, irs].yaxis.set_major_locator(MultipleLocator(dtck))

            if (rs1 == 4 and rs2 == 22) or (rs1 == 22 and rs1 == 4):
                ax[0,irs].set_ylim(-.5,.75)
                ax[1,irs].set_ylim(-.5,.75)

            ax[irs,iax].annotate(f'({lbl_from_idx[2*iax+irs]})',(-.28,.95),
                xycoords = 'axes fraction', annotation_clip = False,fontsize=20
            )

        ax[1,irs].set_xlabel('$q/k_\\mathrm{F}$',fontsize=14)
        ax[0,irs].annotate('$r_\\mathrm{s}=' + f'{rs}$',(0.05,0.875),fontsize=20,xycoords = 'axes fraction')
        
        ax[0,0].legend(fontsize=10, 
            ncol = 1, loc = 'lower right', 
            frameon=False
        )
        
    plt.subplots_adjust(wspace=.3)
    #plt.show() ; exit()
    
    plt.savefig(f'{sdir}/DF-rs_{rs1}_and_{rs2}.pdf',dpi=600,bbox_inches='tight')
    
    plt.cla()
    plt.clf()
    plt.close()
 
    return


def plasmon_plots(rs, fxc_l,
    q_min = 1.e-2, q_max = 3., Nq = 2000,
    use_stored_data = True
):
    x_l = np.linspace(q_min, q_max, Nq)

    fig, ax = plt.subplots(2,1,figsize=(6,6))

    DF_l = ['TCTC','TCTE']

    for fxc in fxc_l:

        for iax, wvar in enumerate(DF_l):

            dfile = f"{plas_dir}/{fxc.replace(' ','-')}/rs_{rs}/wc_{wvar}_rs_{rs}.csv"
            if path.isfile(dfile) and use_stored_data:
                tdat = np.genfromtxt(dfile, delimiter=',', skip_header=1)
            else:
                tdat = get_plasmon_dispersion(x_l, rs, fxc, wvar)

            ax[iax].plot(tdat[:,0], tdat[:,1], 
                color=style_d[fxc][0], linestyle=style_d[fxc][1], 
                label=fxc
            )
    
    for iax, wvar in enumerate(DF_l):
        ax[iax].set_xlim(q_min,q_max)
        ax[iax].hlines(1.,q_min, q_max, color = 'gray', linestyle='-', linewidth=1)
        ax[iax].set_ylabel('$\\omega_\\mathrm{c}^{(\\mathrm{' +wvar + '})}(q)/\\omega_\mathrm{p}(0)$',fontsize=14)

        tmp_x_tcks = ax[iax].get_xticks()
        dtck = tmp_x_tcks[1] - tmp_x_tcks[0]
        ax[iax].xaxis.set_minor_locator(MultipleLocator(dtck/2.))
        ax[iax].xaxis.set_major_locator(MultipleLocator(dtck))

        tmp_y_tcks = ax[iax].get_yticks()
        dtck = tmp_y_tcks[1] - tmp_y_tcks[0]
        ax[iax].yaxis.set_minor_locator(MultipleLocator(dtck/2.))
        ax[iax].yaxis.set_major_locator(MultipleLocator(dtck))

    if abs(rs - 22.) < atol:
        ncol = 3
        lloc = 'lower right'
        bds = [(0., 1.8) for _ in range(2)]
    elif abs(rs - 4.) < atol:
        ncol = 2
        lloc = 'upper left'
        bds = [(.9, 1.6) for _ in range(2)]
    else:
        ncol = 3
        lloc = 'lower right'
        bds = [ax[iax].get_ylim() for iax in range(2)]
    
    for iax in range(2):
        ax[iax].set_ylim(*bds[iax])

    ax[1].set_xlabel('$q/k_\\mathrm{F}$',fontsize=14)
    ax[1].legend(title = '$r_\\mathrm{s}='+f'{rs}$',title_fontsize=10,fontsize=10, 
        ncol = ncol, loc = lloc, 
        frameon=False
    )

    ax[0].annotate('(a)',(-.16,1.),xycoords = 'axes fraction', annotation_clip = False,fontsize=20)
    ax[1].annotate('(b)',(-.16,1.),xycoords = 'axes fraction', annotation_clip = False,fontsize=20)

    
    #plt.show() ; exit()
    plt.savefig(f'{sdir}/plasmon-rs_{rs}.pdf',dpi=600,bbox_inches='tight')
 
    plt.cla()
    plt.clf()
    plt.close()
 
    return

def TCTE_plasmon_pair_plots(fxc_l,
    rs1=4, rs2 = 22,
    q_min = 1.e-2, q_max = 3., Nq = 2000,
    use_stored_data = True
):
    x_l = np.linspace(q_min, q_max, Nq)

    fig, ax = plt.subplots(2,1,figsize=(6,6))
    rs_l = [rs1,rs2]

    #for iax, rs in enumerate(rs_l):
    #    dv = HEG(rs)
    #    ax[iax].plot(x_l, (0.5*x_l**2 - x_l)*dv.kF**2/dv.wp0,color='darkorange',linestyle='--',label='$q^2/2 - q k_\\mathrm{F}$')
    #    ax[iax].plot(x_l, (0.5*x_l**2 + x_l)*dv.kF**2/dv.wp0,color='darkorange',linestyle='-.',label='$q^2/2 + q k_\\mathrm{F}$')

    for iax, rs in enumerate(rs_l):
            
        dv = HEG(rs)

        for fxc in fxc_l:

            dfile = f"{plas_dir}/{fxc.replace(' ','-')}/rs_{rs}/wc_TCTE_rs_{rs}.csv"
            if path.isfile(dfile) and use_stored_data:
                tdat = np.genfromtxt(dfile, delimiter=',', skip_header=1)
            else:
                tdat = get_plasmon_dispersion(x_l, rs, fxc, 'TCTE')

            ph_lbd = (0.5*tdat[:,0]**2 - tdat[:,0])*dv.kF**2/dv.wp0
            ph_ubd = (0.5*tdat[:,0]**2 + tdat[:,0])*dv.kF**2/dv.wp0

            msk = (tdat[:,1] >= ph_ubd) 
            ax[iax].plot(tdat[msk,0], tdat[msk,1], 
                color=style_d[fxc][0], linestyle=style_d[fxc][1], 
                label=fxc
            )
    
    for iax, rs in enumerate(rs_l):
        ax[iax].set_xlim(q_min,q_max)
        ax[iax].hlines(1.,q_min, q_max, color = 'gray', linestyle='-', linewidth=1)
        ax[iax].set_ylabel(r'$\omega_\mathrm{c}(q)/\omega_\mathrm{p}(0)$',fontsize=14)

        tmp_x_tcks = ax[iax].get_xticks()
        dtck = tmp_x_tcks[1] - tmp_x_tcks[0]
        ax[iax].xaxis.set_minor_locator(MultipleLocator(dtck/2.))
        ax[iax].xaxis.set_major_locator(MultipleLocator(dtck))

        tmp_y_tcks = ax[iax].get_yticks()
        dtck = .2#tmp_y_tcks[1] - tmp_y_tcks[0]
        ax[iax].yaxis.set_minor_locator(MultipleLocator(dtck/2.))
        ax[iax].yaxis.set_major_locator(MultipleLocator(dtck))

        if iax == 0:
            mmod = 'a'
        elif iax == 1:
            mmod = 'b'
        ax[iax].annotate(f'({mmod}) ' + '$r_\\mathrm{s}='+f'{rs}$',(.01,.05),fontsize=18,xycoords='axes fraction')

    bds = [(.8, 1.6),(.8, 1.6)]
    
    for iax in range(2):
        ax[iax].set_ylim(*bds[iax])

    ax[1].set_xlabel('$q/k_\\mathrm{F}$',fontsize=14)
    ax[1].legend(fontsize=10, 
        ncol = 3, loc = 'upper left', 
        frameon=False
    )

    #ax[0].annotate('(a)',(-.16,1.),xycoords = 'axes fraction', annotation_clip = False,fontsize=20)
    #ax[1].annotate('(b)',(-.16,1.),xycoords = 'axes fraction', annotation_clip = False,fontsize=20)

    
    #plt.show() ; exit()
    plt.savefig(f'{sdir}/plasmon-rs_{rs1}_and_{rs2}.pdf',dpi=600,bbox_inches='tight')
 
    plt.cla()
    plt.clf()
    plt.close()
 
    return


def W_plots_real_space(rs, fxc_l, 
    fxc_opts = {}, r_min = 0.1, r_max = 5., Nr = 500, Nproc = 1
):

    fig, ax = plt.subplots(2,1,figsize=(5,6))

    _heg = HEG(rs)
    DF_l = ['TCTC','TCTE']

    r_l = np.linspace(r_min, r_max, Nr)
    for iax in range(2):
        ax[iax].plot(r_l,-1./r_l, color = 'darkorange', linestyle = '-', label = 'Bare Coulomb')
    
    for fxc in fxc_l:

        for iax, wvar in enumerate(DF_l):

            if fxc == 'BARE':
                rsfl = './real_space/BARE.csv'
            else:
                rsfl = f"./real_space/{fxc.replace(' ','-')}/rs_{rs}/{fxc}_{wvar}_rs-{rs}.csv"

            if path.isfile(rsfl):
                r_l, wcr = np.transpose(np.genfromtxt(rsfl,delimiter=',',skip_header=1))
            else:
                r_l, wcr = FT_3D_SPH().eval_WCOUL_r(rs, fxc, wvar, 
                    fxc_opts = fxc_opts,
                    r_min = r_min, 
                    r_max = r_max, 
                    Nr = Nr, 
                    Nproc = Nproc
                )

            ax[iax].plot(r_l, -wcr/_heg.kF,
                color=style_d[fxc][0], linestyle=style_d[fxc][1], 
                label=fxc
            )
    
    if rs > 10.:
        ubd = 1.6
    else:
        ubd = 0.5

    for iax, wvar in enumerate(DF_l):
        ax[iax].set_xlim(r_min,r_max)
        ax[iax].hlines(0.,r_min, r_max, color = 'gray', linestyle='-', linewidth=1)

        ax[iax].set_ylim(-3.,ubd)
        ax[iax].set_ylabel('$-W_\\mathrm{' + f'{wvar}'+'}(r)/k_\\mathrm{F}$',fontsize=14)

        tmp_x_tcks = ax[iax].get_xticks()
        dtck = tmp_x_tcks[1] - tmp_x_tcks[0]
        ax[iax].xaxis.set_minor_locator(MultipleLocator(dtck/2.))
        ax[iax].xaxis.set_major_locator(MultipleLocator(dtck))

        tmp_y_tcks = ax[iax].get_yticks()
        dtck = tmp_y_tcks[1] - tmp_y_tcks[0]
        ax[iax].yaxis.set_minor_locator(MultipleLocator(dtck/2.))
        ax[iax].yaxis.set_major_locator(MultipleLocator(dtck))

    ax[1].set_xlabel('$k_\\mathrm{F} \\, r$',fontsize=14)
    ax[1].legend(title = '$r_\\mathrm{s}='+f'{rs}$',title_fontsize=10,fontsize=10, ncol = 2, frameon= False,
        loc = 'lower right'#, bbox_to_anchor = (0.0,1.01)
    )

    ax[0].annotate('(a)',(-.16,.9),xycoords = 'axes fraction', annotation_clip = False,fontsize=20)
    ax[1].annotate('(b)',(-.16,.9),xycoords = 'axes fraction', annotation_clip = False,fontsize=20)
    
    #plt.show() ; exit()

    plt.savefig(f'{sdir}/WCOUL_real_space-rs_{rs}.pdf',dpi=600,bbox_inches='tight')
    
    plt.cla()
    plt.clf()
    plt.close()

    return

def get_flat_plasmons(fxc_l, DF_l = ['TCTC','TCTE']):

    csv_str = 'fxc'
    for _DF in DF_l:
        csv_str += f', {_DF}'
    csv_str += '\n'

    out_d = {}
    for fxc in fxc_l:
        out_d[fxc] = {}
        csv_str += f'{fxc}'
        for _DF in DF_l:
            _ffp = find_flat_plasmon(fxc, _DF,enforce_zeroth=False)
            _ffp.find_rs_crit()
            _ffp.save()
            out_d[fxc][_DF] = _ffp.rsc
            csv_str += f', {_ffp.rsc:.4f}'
        csv_str += '\n'
    
    with open(f'./{plas_dir_for_taylor}/rsc_comp.json','w+') as _logfile_:
        json.dump(out_d,_logfile_)
    
    with open(f'./{plas_dir_for_taylor}/rsc_comp.csv','w+') as _logfile_:
        _logfile_.write(csv_str)
    
    return

def eps_loc_plots(q0,
    fxc_l = ['RPA','ALDA','ALDA-shear'],
    DF_l = ['TCTC','TCTE'],
    rs_min = 0.1,
    rs_max = 30.,
    Nrs = 5000
):

    fig, ax = plt.subplots(2,figsize=(6,4))

    rs_l = np.linspace(rs_min,rs_max,Nrs)
    _HEG_ = HEG(rs_l)

    ybds = [1e20*(-1)**i for i in range(4)]
    for fxc in fxc_l:
        for iax, wvar in enumerate(DF_l):
            IDF = inverse_DF(q0*_HEG_.kF, 1.e-12j*_HEG_.wp0, rs_l, fxc, wvar, fxc_opts = {}).real
            ax[iax].plot(rs_l,IDF, color=style_d[fxc][0], linestyle=style_d[fxc][1], 
                label=fxc)
            ybds[2*iax] = min(ybds[2*iax],IDF.min())
            ybds[2*iax+1] = max(ybds[2*iax+1],IDF.max())

    for iax in range(2):
        ax[iax].set_xlim(rs_min,rs_max)
        ax[iax].set_ylim(ybds[2*iax],ybds[2*iax+1])
        ax[iax].set_ylabel('$\\epsilon^{-1}_{\\mathrm{'+DF_l[iax]+'}}(0,0,r_\\mathrm{s})$')
    
    plt.show()

    return

def statistical_plasmon(rs, fxc_param, fxc_opts = {}, q_min = 0.05, q_max = 4., Nq = 500):

    _HEG_ = HEG(rs)
    moms = []
    for imom in range(3):

        sfl = f'{odir}/{fxc_param}/rs_{rs}/moment_{imom}.csv'
        if path.isfile(sfl):
            ql, tspec = np.transpose(np.genfromtxt(sfl,delimiter=',',skip_header=1))
        else:
            ql, tspec = freq_mom_single_rs(q_min, q_max, Nq, rs, imom, fxc_param, fxc_opts = fxc_opts)
        moms.append(tspec)
    
    avg_w = moms[1]/(moms[0]*_HEG_.wp0)
    dev_w = np.maximum(0., moms[2]/(moms[0]*_HEG_.wp0**2) - avg_w**2)**(0.5)

    fig, ax = plt.subplots(figsize = (6,4))

    ax.plot(ql,avg_w,color='darkblue',label=fxc_param)
    ax.fill_between(ql, avg_w-dev_w, avg_w+dev_w,color='tab:blue', alpha = 0.5)

    cutoff = (ql**2/2. + _HEG_.kF*ql)*_HEG_.kF**2/_HEG_.wp0
    ax.plot(ql,cutoff, color='darkorange', label='$q^2/2 + k_\\mathrm{F} q$')
    
    ax.set_xlim(q_min,q_max)
    ax.set_xlabel('$q/k_\\mathrm{F}$',fontsize=12)

    ax.set_ylabel(r'$\langle\omega(q)\rangle/\omega_\mathrm{p}(0)$',fontsize=12)
    ax.set_ylim(0., 1.02*(avg_w+dev_w).max())
    
    #ax.annotate(f'{fxc_param}, '+'$r_\\mathrm{s}=' + f'{rs}$',(0.02,.9),xycoords='axes fraction',size=18)

    ax.legend(fontsize=12,
        title='$r_\\mathrm{s}=' + f'{rs}$', title_fontsize=14,
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

    #plt.show() ; exit()
    plt.savefig(f'{sdir}stat_plas-fxc_{fxc_param}-rs_{rs}.pdf',dpi=600,bbox_inches='tight')

    plt.cla()
    plt.clf()
    plt.close()

    return


if __name__ == "__main__":

    for rs in [4,22,69]:
        statistical_plasmon(rs, 'AKCK')
    exit()

    #get_flat_plasmons(['ALDA','CDOP','AKCK','Static MCP07','GKI','MCP07','rMCP07'])

    rs_l = [4, 22]

    #DF_pair_plot(4,22,['RPA','ALDA','CDOP','AKCK','Static MCP07']);exit()
    TCTE_plasmon_pair_plots(['RPA','ALDA','CDOP','AKCK','Static MCP07', 'QV spline', 'MCP07', 'rMCP07'],
        q_max = 2
    ) ; exit()

    for ars in rs_l:
        
        W_plots_real_space(ars, 
            ['RPA','ALDA','CDOP','AKCK','Static MCP07'],
            Nproc = 6
        )
        W_plots(ars, ['RPA','ALDA','CDOP','AKCK','Static MCP07'])
        DF_plots(ars, ['RPA','ALDA','CDOP','AKCK','Static MCP07'])

        qmax = 3.
        if abs(ars - 4.) < atol:
            qmax = 1.2
        plasmon_plots(ars, ['RPA','ALDA','CDOP','AKCK','Static MCP07','GKI', 'QV spline', 'MCP07', 'rMCP07'],q_max=qmax)
