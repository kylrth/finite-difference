"""Crank-Nicholson (implicit) finite difference method for heat equation. Code written by Kyle Roth. Implicit finite
difference method derived by Kyle Roth and Michael Nelson.

2018-12-04
"""

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt

# TODO: Turn this into a more general implicit finite difference solver that accepts A and B as arguments (or
# something).

def heat_equation(a, b, T, N_x, N_t, u_0, c_a, d_a, h_a, c_b, d_b, h_b):
    """Performs the Crank Nicolson method for the heat equation. Generally, these implicit methods are written as

        BU^{j+1}=AU^j,

    where 0 <= j < N_t is the time step. Because we may assume that the boundary conditions are satisfied by U^j, we
    multiply A only by the interior of U^j, and then add on a constant term on each end used to set the boundary
    conditions on U^{j+1}.

    More explanation coming soon.
    """
    x, delx = np.linspace(a, b, N_x, retstep=True)
    t, delt = np.linspace(0, T, N_t, retstep=True)

    lamby = delt / delx / delx / 2

    # evaluate the boundary condition functions along x or t
    H_a = h_a(t)
    H_b = h_b(t)
    C_a = c_a(t)
    C_b = c_b(t)
    D_a = d_a(t)
    D_b = d_b(t)
    f_x0 = u_0(x)

    # construct A
    A = np.zeros((N_x - 2, N_x - 2))
    np.fill_diagonal(A, 1 - 2 * lamby)
    np.fill_diagonal(A[1:], lamby)
    np.fill_diagonal(A[:, 1:], lamby)

    # construct B
    B = np.zeros((N_x, N_x))
    np.fill_diagonal(B, 1 + 2 * lamby)
    np.fill_diagonal(B[1:], -lamby)
    np.fill_diagonal(B[:, 1:], -lamby)

    # zero out the top and bottom rows of B
    B[0] = np.zeros(N_x)
    B[-1] = np.zeros(N_x)

    # temporal iteration
    yous = [f_x0]

    for j in range(1, N_t):
        # modify B to leave boundary conditions intact
        B[0, 0] = delx * C_a[j] - D_a[j]
        B[0, 1] = D_a[j]
        B[-1, -1] = delx * C_b[j] + D_b[j]
        B[-1, -2] = -D_b[j]

        # calculate the right hand side
        temp = np.dot(A, yous[-1][1:-1])

        # solve for U^{j+1} on the left after concatenating the right hand side
        yous.append(la.solve(B, np.concatenate(([delx * H_a[j]], temp, [delx * H_b[j]]))))

    return np.array(yous)


def test_heat_flow():
    # Test should produce np.exp(-t) * np.sin(x) as a solution
    """Tests against the analytic solutions for various parameters.

    With initial condition u_0(x) = sin(x) and boundary conditions specified by

        c_a(t) = 1, d_a(t) = 0, h_a(t) = 0,
        c_b(t) = 1, d_b(t) = 0, and h_b(t) = 0,

    the solution is u(x, t) = exp(-t) * sin(x). We test `burgers_equation` using this fact. The correct result is
    displayed as an animation in test_heat_flow.mp4.
    """

    # TODO: fix this. Not working for the following test.

    # boundary condition functions
    Bn = [0, 1, 0.2, 0, -0.12]
    def u0(x, t):
        out = 0.0
        for j, Bn_j in enumerate(Bn):
            out += Bn_j * np.sin(j * np.pi * x) * np.exp(-j ** 2 * np.pi ** 2 * t)
        return out

    def u0_x(x, t):
        out = 0.0
        for j, Bn_j in enumerate(Bn):
            out += Bn_j * j * np.pi * np.cos(j * np.pi * x) * np.exp(-j ** 2 * np.pi ** 2 * t)
        return out

    u_0 = lambda x: u0(x, 0)

    c_a = lambda t: np.zeros_like(t) if isinstance(t, np.ndarray) else 1
    d_a = lambda t: np.ones_like(t) if isinstance(t, np.ndarray) else 0

    c_b = lambda t: np.zeros_like(t) if isinstance(t, np.ndarray) else 1
    d_b = lambda t: np.ones_like(t) if isinstance(t, np.ndarray) else 0

    h_a = lambda t: c_a(t) * u0(a, t) + d_a(t) * u0_x(a, t)
    h_b = lambda t: c_b(t) * u0(b, t) + d_b(t) * u0_x(b, t)

    # conditions
    a = 0.0
    b = 1.0
    T = 0.5
    N_x = 30
    N_t = 30

    soln = heat_equation(a, b, T, N_x, N_t, u_0, c_a, d_a, h_a, c_b, d_b, h_b)

    x = np.linspace(a, b, N_x)

    plt.plot(x, soln[0], label='approx(x, 0)')
    plt.plot(x, soln[14], label='approx(x, 0.25)')
    plt.plot(x, soln[29], label='approx(x, 0.5)')

    plt.legend()
    plt.savefig('test_heat_flow.png')
    plt.close()


if __name__ == '__main__':
    test_heat_flow()
