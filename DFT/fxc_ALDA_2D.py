
import numpy as np

from constants import pi

"""
This module based on the 2D LDA parameterization of 
    C. Attaccalite, S. Moroni, P. Gori-Giorgi, and G.B. Bachelet,
    Phys. Rev. Lett. 88, 256601 (2002),
    DOI: 10.1103/PhysRevLett.88.256601,

    and erratum ibid. 91, 109902 (2003),
    DOI: 10.1103/PhysRevLett.91.109902
"""

def alpha_rs(rs,pars):
    """ Eq. 4 """
    f1 = pars["B"]*rs + pars["C"]*rs**2 + pars["D"]*rs**3
    d_f1_drs = pars["B"] + 2*pars["C"]*rs + 3*pars["D"]*rs**2
    d_f1_drs_2 = 2*pars["C"] + 6*pars["D"]*rs

    f2 = pars["E"]*rs + pars["F"]*rs**(1.5) + pars["G"]*rs**2 + pars["H"]*rs**3
    d_f2_drs = pars["E"] + 3./2.*pars["F"]*rs**(0.5) + 2*pars["G"]*rs + 3*pars["H"]*rs**2
    d_f2_drs_2 = 3./4.*pars["F"]/rs**(0.5) + 2*pars["G"] + 6*pars["H"]*rs

    f3 = np.log(1. + 1./f2)
    f3_den = f2 + f2**2
    d_f3_drs = -d_f2_drs/f3_den
    d_f3_drs_2 = (d_f2_drs**2*(1. + 2*f2)/f3_den - d_f2_drs_2)/f3_den

    alpha = pars["A"] + f1*f3
    d_alpha_drs = d_f1_drs*f3 + f1*d_f3_drs
    d_alpha_drs_2 = d_f1_drs_2*f3 + 2*d_f1_drs*d_f3_drs + f1*d_f3_drs_2
    
    return alpha, d_alpha_drs, d_alpha_drs_2

def eps_x_2D_unp(rs):
    """ Just below Eq. 2 """
    cx = -2**(5./2.)/(3*pi)
    eps_x = cx/rs
    d_eps_x_drs = -eps_x/rs
    d_eps_x_drs_2 = 2.*eps_x/rs**2
    return eps_x, d_eps_x_drs, d_eps_x_drs_2

def eps_c_2D_unp(rs):
    """ Table II """
    unp_pars = {
        "A": -0.1925,
        "B": 0.0863136,
        "C": 0.0572384,
        "E": 1.0022,
        "F": -0.02069,
        "G": 0.33997,
        "H": 1.747e-2
    }
    # caption of Table II indicates constraint on D
    unp_pars["D"] = -unp_pars["A"]*unp_pars["H"]
    return alpha_rs(rs,unp_pars)

def fxc_ALDA_2D(rs):

    _, d_eps_x_drs, d_eps_x_drs_2 = eps_x_2D_unp(rs)
    _, d_eps_c_drs, d_eps_c_drs_2 = eps_c_2D_unp(rs)

    d_eps_xc_drs = d_eps_x_drs + d_eps_c_drs
    d_eps_xc_drs_2 = d_eps_x_drs_2 + d_eps_c_drs_2

    fxc = pi*rs**3*( rs*d_eps_xc_drs_2 - d_eps_xc_drs)/4.
    return fxc