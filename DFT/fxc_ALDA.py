import numpy as np

from constants import pi
from DFT.common import HEG

# From J. P. Perdew and Alex Zunger,
# Phys. Rev. B 23, 5048, 1981
# doi: 10.1103/PhysRevB.23.5048

PZ81U = {
    'A': 0.0311, 'B': -0.048, 'C': 0.0020, 'D': -0.0116, # for rs < 1
    'gamma': -0.1423, 'beta1': 1.0529, 'beta2': 0.3334 # for rs > 1
}

# From J. P. Perdew and W. Yang
# Phys. Rev. B 45, 13244 (1992).
# doi: 10.1103/PhysRevB.45.13244
PW92U = {
    'A': 0.031091,
    'alpha': 0.21370,
    'beta1': 7.5957,
    'beta2': 3.5876,
    'beta3': 1.6382,
    'beta4': 0.49294
}

def fxc_ALDA(dv: HEG,
    x_only: bool = False,
    param: str = 'PW92'
):

    fx = -pi/dv.kF**2
    if x_only:
        return fx

    # The uniform electron gas adiabatic correlation kernel according to
    if param == 'PZ81':
        # Perdew and Zunger, Phys. Rev. B, 23, 5076 (1981)        
        if hasattr(dv.rs,'__len__'):
            fc = np.zeros((dv.rs).shape)
            tmsk = dv.rs < 1.
            fc[tmsk] = PZ81_fc_small_rs(dv.rs[tmsk], dv.n[tmsk])

            tmsk = dv.rs >= 1.
            fc[tmsk] = PZ81_fc_large_rs(dv.rsh[tmsk], dv.n[tmsk])
        
        else:
            if dv.rs < 1.:
                fc = PZ81_fc_small_rs(dv.rs, dv.n)
            else:
                fc = PZ81_fc_large_rs(dv.rsh, dv.n)

    elif param == 'PW92':
        # J. P. Perdew and W. Yang, Phys. Rev. B 45, 13244 (1992).
        q = 2*PW92U['A']*(PW92U['beta1']*dv.rsh + PW92U['beta2']*dv.rs + PW92U['beta3']*dv.rsh**3 + PW92U['beta4']*dv.rs**2)
        dq = PW92U['A']*(PW92U['beta1']/dv.rsh + 2*PW92U['beta2'] + 3*PW92U['beta3']*dv.rsh + 4*PW92U['beta4']*dv.rs)
        ddq = PW92U['A']*(-PW92U['beta1']/2.0/dv.rsh**3 + 3.0/2.0*PW92U['beta3']/dv.rsh + 4*PW92U['beta4'])

        d_ec_d_rs = 2*PW92U['A']*( -PW92U['alpha']*np.log(1.0 + 1.0/q) + (1.0 + PW92U['alpha']*dv.rs)*dq/(q**2 + q) )
        d2_ec_d_rs2 = 2*PW92U['A']/(q**2 + q)*(  2*PW92U['alpha']*dq + (1.0 + PW92U['alpha']*dv.rs)*( ddq - (2*q + 1.0)*dq**2/(q**2 + q) )  )

        fc = dv.rs/(9.0*dv.n)*(dv.rs*d2_ec_d_rs2 - 2*d_ec_d_rs)

    return fx + fc

def PZ81_fc_small_rs(rs, n):
    return -(3*PZ81U['A'] + 2*PZ81U['C']*rs*np.log(rs) + (2*PZ81U['D'] + PZ81U['D'])*rs)/(9*n)

def PZ81_fc_large_rs(rsh, n):
    cons = [5*PZ81U['beta1'],
        7*PZ81U['beta1']**2 + 8*PZ81U['beta2'], 
        21*PZ81U['beta1']*PZ81U['beta2'], 
        (4*PZ81U['beta2'])**2
    ]
    fc_gtr = PZ81U['gamma']/(36*n)/(1. + rsh*(PZ81U['beta1'] + rsh*PZ81U['beta2']))**3 \
        * rsh*(cons[0] + rsh*(cons[1] + rsh*(cons[2] + rsh*cons[3])))
    return fc_gtr

def LDA_derivs(dv : HEG, param : str = 'PW92'):

    if param == 'PZ81':
        eps_c = PZ81U['gamma']/(1.0 + PZ81U['beta1']*dv.rsh + PZ81U['beta2']*dv.rs)
        eps_c_lsr = PZ81U['A']*np.log(dv.rs) + PZ81U['B'] + PZ81U['C']*dv.rs*np.log(dv.rs) + PZ81U['D']*dv.rs

        d_eps_c_d_rs = -PZ81U['gamma']*(0.5*PZ81U['beta1']/dv.rsh + PZ81U['beta2'])\
            /(1.0 + PZ81U['beta1']*dv.rsh + PZ81U['beta2']*dv.rs)**2
        d_ec_drs_lsr = PZ81U['A']/dv.rs + PZ81U['C'] + PZ81U['C']*np.log(dv.rs) + PZ81U['D']

        if hasattr(dv.rs,'__len__'):
            eps_c[dv.rs < 1.0] = eps_c_lsr[dv.rs < 1.0]
            d_eps_c_d_rs[rs < 1.0] = d_ec_drs_lsr[rs < 1.0]
        else:
            if dv.rs < 1.0:
                eps_c = eps_c_lsr[dv.rs < 1.0]
                d_eps_c_d_rs = d_ec_drs_lsr                

    elif param == 'PW92':
        q = 2*PW92U['A']*(PW92U['beta1']*dv.rsh + PW92U['beta2']*dv.rs + PW92U['beta3']*dv.rsh**3 + PW92U['beta4']*dv.rs**2)
        dq = PW92U['A']*(PW92U['beta1']/dv.rsh + 2*PW92U['beta2'] + 3*PW92U['beta3']*dv.rsh + 4*PW92U['beta4']*dv.rs)

        eps_c = -2*PW92U['A']*(1.0 + PW92U['alpha']*dv.rs)*np.log(1.0 + 1.0/q)
        d_eps_c_d_rs = 2*PW92U['A']*( -PW92U['alpha']*np.log(1.0 + 1.0/q) + (1.0 + PW92U['alpha']*dv.rs)*dq/(q**2 + q) )

    else:
        raise SystemExit('Unknown LDA, ',param)

    return eps_c, d_eps_c_d_rs

