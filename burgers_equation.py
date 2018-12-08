# Crank-Nicholson (implicit) finite difference method for Burger's equation.
# Code written by Kyle Roth. Implicit finite difference method derived by Kyle Roth, Michael Nelson, and Jason Gardiner.
# 2018-12-04

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.optimize import fsolve

# TODO: Turn this into a more general implicit finite difference solver that accepts A and B as arguments (or
# something).

def conditions(U1, U0, K1, K2, h, h_aj, c_aj, d_aj, h_bj, c_bj, d_bj):
    """The nonlinear implicit Crank-Nicholson equations for the transformed Burgers' equation.

    out = [
        h h_aj - (h c_aj - d_aj) U1[0] - d_aj U1[1]
        (U1[1]-U0[1]) - K1[-U1[1](U1[2]-U1[0]) - U0[1](U0[2]-U0[0])] - K2[(U1[2]-2*U1[1]+U1[0]) + (U0[2]-2*U0[1]+U0[0])]
        `-.
        (U1[k]-U0[k]) - K1[-U1[k](U1[k+1]-U1[k-1]) - U0[k](U0[k+1]-U0[k-1])] - K2[(U1[k+1]-2*U1[k]+U1[k-1]) + (U0[k+1]-2*U0[k]+U0[k-1])]
        `-.
        (U1[-2]-U0[-2]) - K1[-U1[-2](U1[-1]-U1[-3]) - U0[-2](U0[-1]-U0[-3])] - K2[(U1[-1]-2*U1[-2]+U1[-3]) + (U0[-1]-2*U0[-2]+U0[-3])]
        h h_bj - (h c_bj + d_bj) U1[-1] - d_bj U1[-2]
    ]
    
    Parameters
        U1 (ndarray): The values of U^(n+1)
        U0 (ndarray): The values of U^n
        s (float): wave speed
        K1 (float): first constant in the equations
        K2 (float): second constant in the equations
        TODO: add more parameter descriptions
    
    Returns
        out (ndarray): The residuals (differences between right- and left-hand sides) of the equation, accounting for
                       boundary conditions.
    """
    lhs = U1[1:-1] - U0[1:-1]
    K1_term = K1 * (-U1[1:-1] * (U1[2:] - U1[:-2]) - U0[1:-1] * (U0[2:] - U0[:-2]))
    K2_term = K2 * (U1[2:] - 2 * U1[1:-1] + U1[:-2] + U0[2:] - 2 * U0[1:-1] + U0[:-2])
    rhs = K1_term + K2_term

    # calculate boundary conditions
    a_condition = (h * c_aj - d_aj) * U1[0] + d_aj * U1[1]
    b_condition = (h * c_bj + d_bj) * U1[-1] - d_bj * U1[-2]
    
    # We want to require the interior according to the finite difference method, and the boundary conditions separately.
    return np.concatenate(([h * h_aj - a_condition], lhs - rhs, [h * h_bj - b_condition]))


def conditions_jac(U1, U0, K1, K2, h, h_aj, c_aj, d_aj, h_bj, c_bj, d_bj):
    """The Jacobian of the nonlinear Crank-Nicholson equations for the Burgers' equation.
           _                                                                                _
          | (-h c_aj + d_aj)  (-d_aj)               0                0             0  ...  0 |
          | (K1 U1[1] - K2)   (K1 (U1[0] - U1[2]))  (K1 U1[1] - K2)  0             0  ...  0 |
    jac = | 0  ...  0          `-.                   `-.             `-.           0  ...  0 |
          | 0  ...  0  (K1 U1[k] - K2)  (K1 (U1[k-1] - U1[k+1]))  (K1 U1[k] - K2)  0  ...  0 |
          | 0  ...  0          `-.                   `-.             `-.         `-.  ...  0 |
          | 0  ...            ...          ...           ...    (-d_bj) ... (-h c_bj - d_bj) |
           --                                                                              --
    
    Parameters
        U1 (ndarray): The values of U^(n+1)
        U0 (ndarray): The values of U^n
        s (float): wave speed
        K1 (float): first constant in the equations
        K2 (float): second constant in the equations
        TODO: add more parameter descriptions
    
    Returns
        jac (ndarray): The residuals (differences between right- and left-hand sides) of the equation, accounting for
                       boundary conditions.
    """
    jac = np.zeros((len(U0), len(U0)))

    # fill the main diagonal
    jac[1:-1, 1:-1] = np.diag(1 + K1 * (U1[2:] - U1[:-2]) + 2 * K2)
    jac[0, 0] = d_aj - h * c_aj
    jac[-1, -1] = -d_bj - h * c_bj

    # fill the left off-diagonal
    jac[1:-1, :-2] = jac[1:-1, :-2] + np.diag(-K1 * U[1:-1] - K2)
    jac[-1, -2] = -d_bj

    # fill the right off-diagonal
    jac[1:-1, 2:] = jac[1:-1, 2:] + np.diag(K1 * U1[1:-1] - K2)
    jac[0, 1] = -d_aj

    return jac


def burgers_equation(a, b, T, N_x, N_t, u_0, c_a, d_a, h_a, c_b, d_b, h_b):
    """Takes parameters for the system described in the spec. TODO: flesh out docstring
    """
    if a >= b:
        raise ValueError('a must be less than b')
    if T < 0:
        raise ValueError('T must be greater than or equal to zero')
    if N_x <= 0:
        raise ValueError('N_x must be greater than zero')
    if N_t <= 0:
        raise ValueError('N_t must be greater than zero')
    
    h = (b - a) / (N_x - 1)
    
    x = np.linspace(a, b, N_x)
    t = np.linspace(0, T, N_t)
    
    # evaluate the boundary condition functions along t
    H_a = h_a(t)
    C_a = c_a(t)
    D_a = d_a(t)
    H_b = h_b(t)
    C_b = c_b(t)
    D_b = d_b(t)

    # evaluate the initial condition function
    f_x0 = u_0(x)

    delt = T / (N_t - 1)
    delx = (b - a) / (N_x - 1)
    K1 = delt / 4 / delx
    K2 = delt / 2 / delx / delx
    
    # temporal iteration
    Us = [f_x0]
    
    for j in range(1, N_t):
        Us.append(fsolve(conditions, Us[-1], args=(Us[-1], K1, K2, h, H_a[j], C_a[j], D_a[j], H_b[j], C_b[j], D_b[j])))
    
    return np.array(Us)


if __name__ == '__main__':
    # Try tanh
    u_0 = lambda x: np.ones_like(x)

    a = -1
    b = 1
    T = 1
    N_x = 151
    N_t = 351
    u_0 = lambda x: 1 - np.tanh(x / 2)
    c_a = lambda t: np.ones_like(t) if type(t) == np.ndarray else 1
    d_a = lambda t: np.ones_like(t) if type(t) == np.ndarray else 1
    h_a = lambda t: 1 - np.tanh((a - t) / 2) - 1 / 2 / (np.cosh((a - t) / 2) ** 2)
    c_b = lambda t: np.ones_like(t) if type(t) == np.ndarray else 1
    d_b = lambda t: np.ones_like(t) if type(t) == np.ndarray else 1
    h_b = lambda t: 1 - np.tanh((b - t) / 2) - 1 / 2 / (np.cosh((b - t) / 2) ** 2)

    x = np.linspace(a, b, N_x)
    Us = burgers_equation(a, b, T, N_x, N_t, u_0, c_a, d_a, h_a, c_b, d_b, h_b)

    # animation
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim((x[0], x[-1]))
    ax.set_ylim((-1, 4))

    # correct solution at t=1
    u_1 = lambda x: 1 - np.tanh((x - 1) / 2)

    plt.plot(x, u_0(x))
    plt.plot(x, u_1(x))

    traj, = plt.plot([], [], color='r', alpha=0.5)

    def update(i):
        traj.set_data(x, Us[i])
        return traj

    plt.legend(['$u(x,0)$', '$u(x,1)$', '$u$'])
    ani = animation.FuncAnimation(fig, update, frames=range(len(Us)), interval=25)
    ani.save('test.mp4')
    plt.close()
