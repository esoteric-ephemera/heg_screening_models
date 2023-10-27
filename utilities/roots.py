
def sign(f):
    if f < 0.:
        sgn = -1.
    elif f == 0.:
        sgn = 0.
    elif f > 0.:
        sgn = 1.
    return sgn

def bracket(func, x0, x1, Nstep = 200):
    xl = [x0 + i*(x1 - x0)/(Nstep - 1.) for i in range(Nstep)]
    sgn0 = sign(func(x0))

    a = x0
    for ix in range(1,Nstep):
        b = xl[ix]
        tsgn = sign(func(b))
        if sgn0*tsgn <= 0.:
            break
        sgn0 = tsgn
        a = b

    return sorted([a, b])
    
def bisect(func, a, b, Nstep = 1000, atol = 1.e-10):
    fa = func(a)
    fb = func(b)
    if fa*fb > 0.:
        raise SystemExit(f'Function values f({a}) = {fa}\n  and f({b}) = {fb}\n have same sign!')
    
    sa = sign(fa)
    sb = sign(fb)
    for istep in range(Nstep):
        mid = (a + b)/2.
        fmid = func(mid)
        if abs(fmid) < atol or abs(a - mid) < atol:
            return mid, {'success': True, 'fopt': fmid}
        smid = sign(fmid)
        if smid == sa:
            a = mid
            sa = smid
        else:
            b = mid
            sb = smid

    return mid, {'success': False, 'fopt': fmid}
