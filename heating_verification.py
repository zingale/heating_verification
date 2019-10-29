"""Solve the chaotic damped driven pendulum using the newer SciPy IVP
ODE interfaces.  Passing optional arguments is now done using the
functools partial.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from functools import partial


def H(x):
    return 0.5*np.exp(-(x-0.5)**2/0.1**2) + 0.25*np.exp(-(x-1.0)**2/0.25**2)

def rhs(x, Y, gamma=1.0):
    """
    our system is Y = [rho, u, e] """
    f = np.zeros_like(Y)

    rho = Y[0]
    u = Y[1]
    e = Y[2]

    denom = gamma * (-gamma * e + e + u**2)

    f[0] = H(x)*rho*(gamma - 1)/(u*denom)
    f[1] = -H(x)*(gamma - 1)/denom
    f[2] = H(x)*(-gamma*e + e + u**2)/(u*denom)
    print(x, f, H(x))
    return f

def integrate(Y0, xmax, gamma, dx=0.05):

    r = solve_ivp(partial(rhs, gamma=gamma),
                  (0.0, xmax),
                  Y0,
                  rtol = 1.e-6,
                  atol = 1.e-6,
                  method="BDF",
                  dense_output=True)

    x = np.arange(0.0, xmax, dx)
    return x, r.sol(x)


if __name__ == "__main__":
    gamma = 1.4

    x_max = 2.0

    x, Y = integrate([1.0, 1.0, 1.0], x_max, gamma, dx=0.01)

    plt.plot(x, Y[0,:], label=r"$\rho$")
    plt.plot(x, Y[1,:], label=r"$u$")
    plt.plot(x, Y[2,:], label=r"$e$")
    #HH = H(x)
    #plt.plot(x, HH/HH.max(), label=r"$H$", ls=":")
    plt.legend(frameon=False)
    plt.savefig("verification.png")
