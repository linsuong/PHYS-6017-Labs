import numpy as np
import matplotlib.pyplot as plt
import Projects.ForcedSimplePendulum.rungekutta as rungekutta
from scipy.integrate import odeint

#SOLVING THE MATRIX EQUATION Y' = AY + b
#parameters:
def alpha(k, m, g, L):
    return (k/(m * L * np.sqrt(g * L)))

def beta(F, m, g):
    return (F/(m * g))

def matrixA(alpha):
    [[alpha, 0],
     [1, 0]]
    
def bvector():
    [[-np.sin(nu) + beta * np.cos((1 - eta) * tau)],
     [0]]

#different method, using inbuilt ODEINT    
def oscillation(theta, t, F, m, L, k, g, omega, eta):
    #parameters:
    alpha = (k/((m * L) * np.sqrt(g * L)))
    beta = F / (m * g)
    
    return theta[1], - alpha * theta[0] - np.sin(theta) + beta * np.cos((1 - eta) * t)

def equation_of_motion(theta, t, m, k, l, F, L, omega):
    g = 9.81
    
    return (m * L * theta[1], -k * theta[1] -m * g * l * np.sin(theta[0]) + F * L * np.cos(omega * t))
    
theta_0 = [0, 0]

t_s = np.linspace(1, 10, 200)

u_s = odeint(equation_of_motion, theta_0, t_s, args= (0.5, 0.1, 0.5, 3, 2, 5))

y_s = u_s[:, 0]

plt.plot(t_s, y_s, 'o')

plt.show()

#print(rk45.rk45(theta, t, F, m, L, k,))

"""
def update(frame):
    line.set_data()
    scatter.set_data()
    
    return (line, scatter)
"""