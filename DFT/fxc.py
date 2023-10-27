
from DFT.fxc_ALDA_2D import fxc_ALDA_2D
from DFT.fxc_DPGT_2D import fxc_DPGT_2D

from DFT.fxc_ALDA import fxc_ALDA
from DFT.fxc_AKCK import fxc_AKCK
from DFT.fxc_CDOP import fxc_CDOP
from DFT.fxc_DLDA import fxc_AKCK_dyn
from DFT.fxc_GKI import fxc_GKI_real_freq
from DFT.fxc_MCP07 import fxc_MCP07_static, fxc_MCP07, fxc_rMCP07
from DFT.fxc_QV import fxc_QV, fxc_QV_spline, fxcl_static

def get_fxc(q, omega, dv, fxc_param, fxc_opts = {}, d = 3):
    if d == 2:
        _fxc = _get_fxc_2D
    elif d == 3:
        _fxc = _get_fxc_3D
    else:
        raise SystemExit(f"Cannot compute fxc in {d} dimensions!")
    return _fxc(q, omega, dv, fxc_param, fxc_opts = fxc_opts)

def _get_fxc_2D(q, omega, dv, fxc_param, fxc_opts = {}):
    """ For consistency with 3D implementation, keep omega and fxc_opts args """

    if fxc_param == "RPA":
        fxc = 0.
    elif fxc_param == "ALDA":
        fxc = fxc_ALDA_2D(dv.rs)
    elif fxc_param == "DPGT":
        fxc = fxc_DPGT_2D(q,dv)
    else:
        raise SystemExit(f"Unknown 2D fxc, {fxc_param}!")

    return fxc

def _get_fxc_3D(q, omega, dv, fxc_param, fxc_opts = {}):

    # local in space, local in time
    if fxc_param == 'RPA':
        fxc = 0.
    elif fxc_param == 'ALDA':
        fxc = fxc_ALDA(dv, x_only = False, param = 'PW92')
    elif fxc_param == 'ALDA-shear':
        fxc = fxcl_static(dv)

    # nonlocal in space, local in time
    elif fxc_param == 'AKCK':
        fxc = fxc_AKCK(q,dv)
    elif fxc_param == 'CDOP':
        fxc = fxc_CDOP(q,dv)
    elif fxc_param in ['Static MCP07', 'MCP07-static']:
        fxc = fxc_MCP07_static(q, dv, param = 'PZ81', kernel_only = True)
    
    # local in space, nonlocal in time
    elif fxc_param == 'GKI':
        fxc = fxc_GKI_real_freq(omega.real, dv, param = 'PW92', revised = True)
    elif fxc_param == 'QV':
        fxc = fxc_QV(omega.real, dv)
    elif fxc_param == 'QV spline':
        fxc = fxc_QV_spline(fxc_opts).eval(omega.real)
    
    # nonlocal in space, nonlocal in time
    elif fxc_param == 'MCP07':
        fxc = fxc_MCP07(q, omega.real, dv)
    elif fxc_param == 'rMCP07':
        fxc = fxc_rMCP07(q, omega.real, dv)
    elif fxc_param == 'AKCK dynamic':
        fxc = fxc_AKCK_dyn(q,omega.real,dv)

    else:
        raise SystemExit(f"Unknown 3D fxc, {fxc_param}!")

    return fxc
