# Crank-Nicholson (implicit) finite difference method for Burger's equation.
# Code written by Kyle Roth. Implicit finite difference method derived by Kyle Roth, Michael Nelson, Jason Gardiner, and
# Jared Nielsen. 2018-12-04

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from matplotlib import animation

# TODO: Turn this into a more general implicit finite difference solver that accepts A and B as arguments (or
# something).

def conditions(U1, U0, K1, K2, h, h_aj, c_aj, d_aj, h_bj, c_bj, d_bj):
    """The nonlinear implicit Crank-Nicholson equations for the transformed Burgers' equation, derived using forward
    difference approximations for u_x and center difference for u_xx. Boundary conditions were derived similarly.

    out = [
        h h_aj - (h c_aj - d_aj) U1[0] - d_aj U1[1]
        (U1[1]-U0[1]) - K1[-U1[1](U1[2]-U1[0]) - U0[1](U0[2]-U0[0])] - K2[(U1[2]-2*U1[1]+U1[0]) + (U0[2]-2*U0[1]+U0[0])]
        `-.
        (U1[k]-U0[k]) - K1[-U1[k](U1[k+1]-U1[k-1]) - U0[k](U0[k+1]-U0[k-1])]  # cont'd
                - K2[(U1[k+1]-2*U1[k]+U1[k-1]) + (U0[k+1]-2*U0[k]+U0[k-1])]
        `-.
        (U1[-2]-U0[-2]) - K1[-U1[-2](U1[-1]-U1[-3]) - U0[-2](U0[-1]-U0[-3])]  # cont'd
                - K2[(U1[-1]-2*U1[-2]+U1[-3]) + (U0[-1]-2*U0[-2]+U0[-3])]
        h h_bj - (h c_bj + d_bj) U1[-1] - d_bj U1[-2]
    ]
    
    Parameters
        U1 (ndarray): The values of U^(n+1)
        U0 (ndarray): The values of U^n
        K1 (float): first constant in the equations
        K2 (float): second constant in the equations
        h (float): spatial difference constant, usually (b - a) / num_x_steps
        h_aj (float): h_a evaluated at this time step
        c_aj (float): c_a evaluated at this time step
        d_aj (float): d_a evaluated at this time step
        h_bj (float): h_b evaluated at this time step
        c_bj (float): c_b evaluated at this time step
        d_bj (float): d_b evaluated at this time step
    
    Returns
        out (ndarray): The residuals (differences between right- and left-hand sides) of the equation, accounting for
                       boundary conditions.
    """
    # compute Crank-Nicolson conditions on interior
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
        K1 (float): first constant in the equations
        K2 (float): second constant in the equations
        h (float): spatial difference constant, usually (b - a) / num_x_steps
        h_aj (float): h_a evaluated at this time step
        c_aj (float): c_a evaluated at this time step
        d_aj (float): d_a evaluated at this time step
        h_bj (float): h_b evaluated at this time step
        c_bj (float): c_b evaluated at this time step
        d_bj (float): d_b evaluated at this time step
    
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
    jac[1:-1, :-2] = jac[1:-1, :-2] + np.diag(-K1 * U1[1:-1] - K2)
    jac[-1, -2] = -d_bj

    # fill the right off-diagonal
    jac[1:-1, 2:] = jac[1:-1, 2:] + np.diag(K1 * U1[1:-1] - K2)
    jac[0, 1] = -d_aj

    return jac


def newton(f, x0, Df, tol=1e-5, maxiters=30, alpha=1., args=()):
    """Use Newton's method to approximate a zero of the function f.

    Parameters:
        f (lambda): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (lambda): The derivative of f, a function from R^n to R^(n*n).
        tol (float): Convergence tolerance. The function should return when the difference between successive
                     approximations is less than tol.
        maxiters (int): The maximum number of iterations to compute.
        alpha (float): Backtracking scalar (Problem 3).

    Returns:
        (float or ndarray): The approximation for a zero of f.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    i = 0
    if np.isscalar(x0):
        while i < maxiters:
            fut = x0 - alpha * f(x0, *args) / Df(x0, *args)
            i += 1
            if abs(fut - x0) < tol:
                return fut, True, i
            else:
                x0 = fut
        return x0, False, i
    else:
        while i < maxiters:
            fut = x0 - alpha * np.linalg.solve(Df(x0, *args), f(x0, *args))
            i += 1
            if np.linalg.norm(fut - x0) < tol:
                return fut, True, i
            else:
                x0 = fut
        return x0, False, i


def burgers_equation(a, b, T, N_x, N_t, u_0, c_a, d_a, h_a, c_b, d_b, h_b):
    """Crank-Nicolson approximation of the solution u(x, t) for the following system:

        u_t + (u ** 2 / 2)_x = u_xx,   a <= x <= b, 0 < t <= T
            u(x, 0) = u_0(x),
            h_a(t) = c_a(t) * u(a, t) + d_a(t) * u_x(a, t),
            h_b(t) = c_b(t) * u(b, t) + d_b(t) * u_x(b, t).

    Parameters:
        a (float): left spatial endpoint
        b (float): right spatial endpoint
        T (float): final time value
        N_x (int): number of mesh nodes in the spatial dimension
        N_t (int): number of mesh nodes in the temporal dimension
        u_0 (callable): function specifying the initial condition
        c_a (callable): function specifying left boundary condition
        d_a (callable): function specifying left boundary condition
        h_a (callable): function specifying left boundary condition
        c_b (callable): function specifying right boundary condition
        d_b (callable): function specifying right boundary condition
        h_b (callable): function specifying right boundary condition
    
    Returns:
        Us (np.ndarray): finite difference approximation of u(x,t). Us[j] = u(x,t_j), where j is the index corresponding
                         to time t_j.
    """
    if a >= b:
        raise ValueError('a must be less than b')
    if T <= 0:
        raise ValueError('T must be greater than or equal to zero')
    if N_x <= 2:
        raise ValueError('N_x must be greater than zero')
    if N_t <= 1:
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
        result, converged, _ = newton(conditions,
                                      Us[-1],
                                      conditions_jac,
                                      args=(Us[-1], K1, K2, h, H_a[j], C_a[j], D_a[j], H_b[j], C_b[j], D_b[j])
                                     )
        if not converged:
            print('warning: Newton\'s method did not converge')
        Us.append(result)

        # Use the following code instead of the above to solve using scipy.optimize.fsolve
        # from scipy.optimize import fsolve
        # Us.append(fsolve(conditions,
        #                  Us[-1],
        #                  args=(Us[-1], K1, K2, h, H_a[j], C_a[j], D_a[j], H_b[j], C_b[j], D_b[j])))
    
    return np.array(Us)


def test_burgers_equation():
    """With initial condition u_0(x) = 1 - tanh(x / 2) and boundary conditions specified by

           c_a(t) = 1, d_a(t) = 1, h_a(t) = 1 - tanh((a - t) / 2) - 0.5 * sech^2((a - t) / 2),
           c_b(t) = 1, d_b(t) = 1, and h_b(t) = 1 - tanh((b - t) / 2) - 0.5 * sech^2((b - t) / 2),

       the solution is u(x, t)= 1 - tanh((x - t) / 2). We test `burgers_equation` using this fact. The correct result is
       displayed as an animation in test_burgers_equation.mp4.
    """
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
    ax.set_ylim((0, 3))

    # correct solution at t=1
    u_1 = lambda x: 1 - np.tanh((x - 1) / 2)

    plt.plot(x, u_0(x))
    plt.plot(x, u_1(x))

    traj, = plt.plot([], [], color='r', alpha=0.5)

    def update(i):
        traj.set_data(x, Us[i])
        return traj

    plt.legend(['theoretical $u(x,0)$', 'theoretical $u(x,1)$', 'approximated $u(x,t)$'])
    ani = animation.FuncAnimation(fig, update, frames=range(len(Us)), interval=25)
    ani.save('test_burgers_equation.mp4')
    plt.close()


if __name__ == '__main__':
    test_burgers_equation()
