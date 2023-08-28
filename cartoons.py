import matplotlib.pyplot as plt
import numpy as np

font = {'family': 'serif', 'serif': ['Palatino']}
plt.rc('font', **font)
plt.rc('text', usetex=True)

def sin(x, a, b, c):
    return a*np.sin(2.*np.pi*(b*x + c))

fig, ax = plt.subplots(2,2,figsize=(6,6))

amp = .65

xl = np.linspace(0.,2.,5000)

plot_pars = {
    0 : {
        0 : {'label': ('a', 2,'anti-'), 'curves': [(amp,1., 0.), (2.*amp,1., 0.)]},
        1 : {'label': ('b', 1, 'no '), 'curves': [(amp,1., 0.), (amp,1., 0.)]}
    },
    1 : {
        0 : {'label': ('c', 0, 'perfect '), 'curves': [(amp,1., 0.), (0., 1., 0.)]},
        1 : {'label': ('d',-1,'over-'), 'curves': [(amp,1., 0.), (amp,1., 0.5)]}
    }
}

colors = ['darkblue','darkorange']
linestyles = ['-','--']

for iax in range(2):
    for jax in range(2):
        for icurve in range(2):
            ax[iax,jax].plot(xl, sin(xl,*plot_pars[iax][jax]['curves'][icurve]),
                linestyle=linestyles[icurve], color=colors[icurve]
            )
        ax[iax,jax].set_xlim(xl.min(),xl.max())
        ax[iax,jax].hlines(0.,xl.min(),xl.max(),color='gray',linestyle='-',linewidth=1,zorder=-1)

        ax[iax,jax].set_ylim(-2.,2.)

        panel_label = '({:}) $\\epsilon^{{-1}}(q,0) = {:}$,\n{:}screening'\
            .format(*plot_pars[iax][jax]["label"])
        ax[iax,jax].annotate(panel_label,(0.05,-1.9),fontsize=12,color='k')
        ax[iax,jax].axis('off')

    ax[1,iax].set_xlabel('$q/\\mathrm{k}_F$',fontsize=14)    

plt.subplots_adjust(hspace=-0.1)
#plt.show() ; exit()
plt.savefig('./figs/eps_cartoons.pdf',dpi=600,bbox_inches='tight')