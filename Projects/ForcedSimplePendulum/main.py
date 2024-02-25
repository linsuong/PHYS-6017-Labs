import numpy as np
import matplotlib.pyplot as plt
import rk45
from scipy.integrate import odeint

def oscillation(theta, t, F, m, L, k, g, omega, eta):
    
    #parameters:
    alpha = (k/((m * L) * np.sqrt(g * L)))
    beta = F / (m * g)
    eta = eta
    
        
    return

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