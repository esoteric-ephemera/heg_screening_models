import numpy as np

pi = np.pi

tol = 7.0/3.0 - 4.0/3.0 - 1 # machine epsilon
for i in range(2000,4000):
    t = np.array([1.0,i])*pi/(i + 1.0)
    sarg = np.sin(t)
    carg = np.cos(t)
    tmp = 1.0 + 2.0/pi*(1 + 2/3*sarg**2)*carg*sarg - 2*t/pi

    if abs(tmp[0] - 1) < tol or abs(tmp[-1] + 1) < tol:
        maxN = i-1
        break

def mod_gauss_chebyshev(N):

    """
        Adapted from J.M. Pérez-Jordá, E. San-Fabián, and F. Moscardó,
            Comp. Phys. Comm. 70, 271 (1992).
            doi: 10.1016/0010-4655(92)90192-2
    """
    errstr = 'Number of quadrature points exceeds permitted maximum {:},\nprecision loss anticipated'.format(maxN)
    assert N <= maxN, errstr
    np1 = N+1.0
    fac = pi/np1
    arg = np.arange(1,N+1,1)*fac
    sarg = np.sin(arg)
    carg = np.cos(arg)

    # Eqs. 13 and 14
    x = 1.0 + 2.0/pi*(1 + 2/3*sarg**2)*carg*sarg - 2*arg/pi
    wg = 16.0/(3.0*np1)*sarg**4

    return x, wg


def m4_grid(Nr,zeta=1.0,alpha=0.6):

    """
        Adapted from O. Treutler and R. Ahlrichs,
            J. Chem. Phys. 102, 346 (1995).
            doi: 10.1063/1.469408
    """
    x, xwg = mod_gauss_chebyshev(Nr)

    opxa = (1.0 + x)**alpha
    fac = zeta/np.log(2.0)
    t1 = np.log(2.0/(1.0 - x))
    r = fac * opxa * t1
    dr = fac * opxa * (alpha/(1.0 + x)*t1 + 1.0/(1.0 - x) )

    #rwg = 4*pi*r**2*dr*xwg
    rwg = dr*xwg

    return r, rwg

if __name__ == "__main__":

    for n in [100,1000,2000]:
        r, _ = m4_grid(n)
        print(n,r.max())
