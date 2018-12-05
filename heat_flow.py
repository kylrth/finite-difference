# Crank-Nicholson (implicit) finite difference method for heat equation.
# Code written by Kyle Roth. Implicit finite difference method derived by Kyle Roth and Michael Nelson.
# 2018-12-04

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
    h = (b - a) / (N_x - 1)
    k = T / (N_t - 1)
    
    lamby = k / h / h / 2
    
    x = np.linspace(a, b, N_x)
    t = np.linspace(0, T, N_t + 1)
    
    # evaluate the boundary condition functions along x or t
    H_a = h_a(x)
    H_b = h_b(x)
    C_a = c_a(t)
    C_b = c_b(t)
    D_a = d_a(t)
    D_b = d_b(t)
    f_x0 = u_0(x)
    
    # construct A
    A = np.zeros((N_x - 2, N_x - 2))
    np.fill_diagonal(A, 1 - 2 * lamby)
    np.fill_diagonal(A[1:], lamby)
    np.fill_diagonal(A[:,1:], lamby)
    
    # construct B
    B = np.zeros((N_x, N_x))
    np.fill_diagonal(B, 1 + 2 * lamby)
    np.fill_diagonal(B[1:], -lamby)
    np.fill_diagonal(B[:,1:], -lamby)
    
    # zero out the top and bottom rows of B
    B[0] = np.zeros(N_x)
    B[-1] = np.zeros(N_x)
    
    # temporal iteration
    yous = [f_x0]
    
    for j in range(1, N_t):
        # modify B to leave boundary conditions intact
        B[0, 0] = h * C_a[j] - D_a[j]
        B[0, 1] = D_a[j]
        B[-1, -1] = h * C_b[j] + D_b[j]
        B[-1, -2] = -D_b[j]
        
        # calculate the right hand side
        temp = np.dot(A, yous[-1][1:-1])
        
        # solve for U^{j+1} on the left after concatenating the right hand side
        yous.append(la.solve(B, np.concatenate(([h * H_a[j]], temp, [h * H_b[j]]))))
    
    return np.array(yous)
