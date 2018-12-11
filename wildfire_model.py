# Crank-Nicholson (implicit) finite difference method for a wildfire model.
# Code written by Kyle Roth. Implicit finite difference method derived by Kyle Roth, Michael Nelson, Jason Gardiner, and
# Jared Nielsen. 2018-12-10

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from matplotlib import animation
from scipy.optimize import fsolve

def conditions(ST1, ST0,  # vectors
               K1, K2, h, k, A, B, C1, C2,  # constants
               hT_aj, cT_aj, dT_aj, hT_bj, cT_bj, dT_bj, hS_aj, cS_aj, dS_aj, hS_bj, cS_bj, dS_bj):  # functions
    """Return the conditions for the wildfire model.
    
    Returns nonlinear implicit Crank-Nicholson conditions for the wildfire PDE system, derived using center difference
    approximations for u_x and midpoint approximation for u_xx. Boundary conditions were derived similarly.

    With K1 = k / (2 * h ** 2) and K2 = k * V / (4 * h), the conditions are the following:

    for T: [
        h hT_aj = (h cT_aj - dT_aj) T1[0] + dT_aj T1[1]  # left boundary
        `-.
        (T1[k] - T0[k]) = 
            K1 * (T1[k+1] - T1[k] + T1[k-1] + T0[k+1] - T0[k] + T0[k-1])
            - K2 * (T1[k+1] - T1[k-1] + T0[k+1] - T0[k-1])  # interior
        `-.
        h hT_bj = (h cT_bj + dT_bj) T1[-1] + dT_bj T1[-2]  # right boundary
    ], and

    for S: [
        h hS_aj = (h cS_aj - dS_aj) S1[0] + dS_aj S1[1]  # left boundary
        `-.
        S1[k] - S0[k] = -k * C2 * S1[k] * exp(-B / T1[k]))  # interior
        `-.
        h hS_bj = (h cS_bj + dS_bj) S1[-1] + dS_bj S1[-2]  # right boundary
    ]
    
    Parameters
        ST1 (ndarray): The values of S^{n+1} and T^{n+1}
        ST0 (ndarray): The values of S^n and T^n
        K1 (float): first constant in the equations
        K2 (float): second constant in the equations
        h (float): spatial difference constant, usually (b - a) / num_x_steps
        k (float): temporal difference constant, usually T / num_t_steps
        A (float): constant from PDE system
        B (float): constant from PDE system
        C1 (float): constant from PDE system
        C2 (float): constant from PDE system
        hT_aj (float): hT_a evaluated at this time step
        cT_aj (float): cT_a evaluated at this time step
        dT_aj (float): dT_a evaluated at this time step
        hT_bj (float): hT_b evaluated at this time step
        cT_bj (float): cT_b evaluated at this time step
        dT_bj (float): dT_b evaluated at this time step
        hS_aj (float): hS_a evaluated at this time step
        cS_aj (float): cS_a evaluated at this time step
        dS_aj (float): dS_a evaluated at this time step
        hS_bj (float): hS_b evaluated at this time step
        cS_bj (float): cS_b evaluated at this time step
        dS_bj (float): dS_b evaluated at this time step
    
    Returns
        (ndarray): The residuals (differences between right- and left-hand sides) of the conditions.
    """
    T0, S0 = np.split(ST0, 2)
    T1, S1 = np.split(ST1, 2)

    # compute Crank-Nicolson conditions on interior for T
    T_lhs = T1[1:-1] - T0[1:-1]
    K1_term = K1 * (T1[2:] - T1[1:-1] + T1[:-2] + T0[2:] - T0[1:-1] + T0[:-2])
    K2_term = K2 * (T1[2:] - T1[:-2] + T0[2:] - T0[:-2])
    T_rhs = K1_term - K2_term + A(S1[1:-1] * np.exp(-B / T1[1:-1]) - C1 * T1[1:-1])

    # calculate boundary conditions for T
    Ta_condition = (h * cT_aj - dT_aj) * T1[0] + dT_aj * T1[1]
    Tb_condition = (h * cT_bj + dT_bj) * T1[-1] - dT_bj * T1[-2]

    # compute Crank-Nicolson conditions on interior for S
    S_lhs = S1[1:-1] - S0[1:-1]  # S1[k] - S0[k] = -k * C2 * S1[k] * exp(-B / T1[k]))
    S_rhs = -k * C2 * S1[1:-1] * np.exp(-B / T1[1:-1])

    # calculate boundary conditions for S
    Sa_condition = (h * cS_aj - dS_aj) * S1[0] + dS_aj * S1[1]
    Sb_condition = (h * cS_bj + dS_bj) * S1[-1] - dS_bj * S1[-2]

    # return the complete set of conditions for S and T
    return np.concatenate((
        [h * hT_aj - Ta_condition],  # T boundary condition at a
        T_lhs - T_rhs,               # T interior conditions
        [h * hT_bj - Tb_condition],  # T boundary condition at b
        [h * hS_aj - Sa_condition],  # S boundary condition at a
        S_lhs - S_rhs,               # S interior conditions
        [h * hS_bj - Sb_condition]   # S boundary condition at b
        ))


def wildfire_model(a, b, T, N_x, N_t,  # constants
                   T_0, S_0, cT_a, dT_a, hT_a, cT_b, dT_b, hT_b, cS_a, dS_a, hS_a, cS_b, dS_b, hS_b,  # functions
                   A, B, C1, C2, v):  # constants
    """Returns a solution to the wildfire PDE system.
    
    Returns a Crank-Nicolson approximation of the solution T(x, t), S(x, t) for the following system:

        T_t = T_xx - v * T_x + A(S * exp(-B / T) - C1 * T),
        S_t = -C2 * S * exp(-B / T),                          a <= x <= b, 0 < t <= T
            T(x, 0) = T_0(x),
            S(x, 0) = S_0(x),
            hT_a(t) = cT_a(t) * T(a, t) + dT_a(t) * T_x(a, t),
            hT_b(t) = cT_b(t) * T(b, t) + dT_b(t) * T_x(b, t),
            hS_a(t) = cS_a(t) * S(a, t) + dS_a(t) * S_x(a, t),
            hS_b(t) = cS_b(t) * S(b, t) + dS_b(t) * S_x(b, t).
        
    In the above equations, T corresponds to temperature, S to the amount of available fuel, and v to wind conditions;
    A, B, C1, and C2 are constants.

    Parameters:
        a (float): left spatial endpoint
        b (float): right spatial endpoint
        T (float): final time value
        N_x (int): number of mesh nodes in the spatial dimension
        N_t (int): number of mesh nodes in the temporal dimension
        T_0 (callable): function specifying the initial condition for T
        S_0 (callable): function specifying the initial condition for S
        cT_a (callable): function specifying left boundary condition for T
        dT_a (callable): function specifying left boundary condition for T
        hT_a (callable): function specifying left boundary condition for T
        cT_b (callable): function specifying right boundary condition for T
        dT_b (callable): function specifying right boundary condition for T
        hT_b (callable): function specifying right boundary condition for T
        cS_a (callable): function specifying left boundary condition for S
        dS_a (callable): function specifying left boundary condition for S
        hS_a (callable): function specifying left boundary condition for S
        cS_b (callable): function specifying right boundary condition for S
        dS_b (callable): function specifying right boundary condition for S
        hS_b (callable): function specifying right boundary condition for S
        A (float): constant from PDE system
        B (float): constant from PDE system
        C1 (float): constant from PDE system
        C2 (float): constant from PDE system
        v (float): constant from PDE system
    
    Returns:
        Ts (np.ndarray): finite difference approximation of T(x,t). Ts[j] = T(x,t_j), where j is the index corresponding
                         to time t_j.
        Ss (np.ndarray): finite difference approximation of T(x,t). Ts[j] = T(x,t_j), where j is the index corresponding
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
    HT_a = hT_a(t)
    CT_a = cT_a(t)
    DT_a = dT_a(t)
    HT_b = hT_b(t)
    CT_b = cT_b(t)
    DT_b = dT_b(t)

    HS_a = hS_a(t)
    CS_a = cS_a(t)
    DS_a = dS_a(t)
    HS_b = hS_b(t)
    CS_b = cS_b(t)
    DS_b = dS_b(t)

    # evaluate the initial condition functions
    T_x0 = T_0(x)
    S_x0 = S_0(x)

    delt = T / (N_t - 1)
    delx = (b - a) / (N_x - 1)
    K1 = delt / 2 / delx / delx
    K2 = delt * v / 4 / delx
    
    # combine the initial conditions for T and S into one vector
    STs = [np.concatenate((T_x0, S_x0))]
    
    for j in range(1, N_t):
        STs.append(fsolve(conditions,
                          STs[-1],
                          args=(STs[-1],
                                K1, K2, h, k, A, B, C1, C2,
                                HT_a[j], CT_a[j], DT_a[j], HT_b[j], CT_b[j], DT_b[j],
                                HS_a[j], CS_a[j], DS_a[j], HS_b[j], CS_b[j], DS_b[j]
                               )
                         ))
    STs = np.array(STs)
    print(STs.shape)
    Ts, Ss = np.split(np.array(STs), 2, axis=1)
    print(Ts.shape, Ss.shape)
    return Ts, Ss


def test_wildfire_model():
    """With initial conditions
    
        T_0(x) = 1 - tanh(x / 2)  # not true
        S_0(x) = 1
    
    and boundary conditions specified by

        c_a(t) = 1, d_a(t) = 1, h_a(t) = 1 - tanh((a - t) / 2) - 0.5 * sech^2((a - t) / 2),  # not true
        c_b(t) = 1, d_b(t) = 1, and h_b(t) = 1 - tanh((b - t) / 2) - 0.5 * sech^2((b - t) / 2),  # not true

    the solution is u(x, t)= 1 - tanh((x - t) / 2)  # not true. We test `burgers_equation` using this fact. The correct result is
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
    test_wildfire_model()
