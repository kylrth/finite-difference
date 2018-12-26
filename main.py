import numpy as np
import matplotlib.pyplot as plt

import wildfire_model as code

'''
-------------------------------------------------------------------------------
'''

kappa = 0.5

A = 1.0
B = 0.2
C1 = 1.0
C2 = 1.5
nu = 1.0


'''
------------------
'''
def sech(x):
    return (1/np.cosh(x))

# solution
def S0(x,t):
    return 1+np.tanh(x)

# solution
def T0(x,t):
    return sech(x)

############################################

a = -5.0
b = 10.0
end_time = 1.0
N_t = 5

# boundary conditions

cS_a = lambda t: np.ones_like(t) if type(t) == np.ndarray else 1.0
dS_a = lambda t: np.zeros_like(t) if type(t) == np.ndarray else 0.0
cS_b = lambda t: np.ones_like(t) if type(t) == np.ndarray else 1.0
dS_b = lambda t: np.zeros_like(t) if type(t) == np.ndarray else 0.0

cT_a = lambda t: np.ones_like(t) if type(t) == np.ndarray else 1.0
dT_a = lambda t: np.zeros_like(t) if type(t) == np.ndarray else 0.0
cT_b = lambda t: np.ones_like(t) if type(t) == np.ndarray else 1.0
dT_b = lambda t: np.zeros_like(t) if type(t) == np.ndarray else 0.0

# dependent boundary conditions
hS_b = lambda t: np.zeros_like(t) if type(t) == np.ndarray else 0.0
hS_a = lambda t: np.zeros_like(t) if type(t) == np.ndarray else 0.0
S_0 = lambda x: S0(x,0)

hT_b = lambda t: np.zeros_like(t) if type(t) == np.ndarray else 0.0
hT_a = lambda t: np.zeros_like(t) if type(t) == np.ndarray else 0.0
T_0 = lambda x: T0(x,0)

num_runs = 3
for index in range(0,num_runs):
    N_t = 2*N_t
t = np.linspace(0,end_time,N_t)
delt = t[1]-t[0]
new_delx = np.sqrt(delt/kappa)
N_x = int(np.round((b-a)/new_delx))
x = np.linspace(a,b,N_x)

T,S = code.wildfire_model(a,b,end_time,N_x,N_t,T_0,S_0,cT_a,dT_a,hT_a,cT_b,
dT_b,hT_b,cS_a,dS_a,hS_a,cS_b,dS_b,hS_b,A,B,C1,C2,nu)

plt.plot(x,T[-1])
plt.plot(x,S[-1])

plt.show()
