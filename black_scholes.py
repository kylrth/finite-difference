# Estimation of the solution to the Black-Scholes equation using the Euler-Maruyama method.
# Code written by Kyle Roth.
# 2018-12-17


import numpy as np


def euler_maruyama(t, x_0, mu, sigma):
    """Return an estimate of the solution to the Black-Scholes equation using the Euler-Maruyama method, which is
    essentially a finite difference method.

    Mu and sigma are the drift and volatility of the system over one time step.
    
    Parameters:
        t (ndarray): set of time steps to estimate at
        x_0 (float): initial value of the equation
        mu (float): daily drift constant.
    """
    del_t = t[1:] - t[:-1]
    del_wt = np.random.normal(scale=np.sqrt(del_t))

    del_x = mu * del_t + sigma * del_wt

    # add on x_0 to start the cumulative sum
    del_x = np.concatenate(([x_0], del_x))
    
    return np.cumsum(del_x)
