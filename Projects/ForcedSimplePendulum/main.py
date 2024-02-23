import numpy as np
import matplotlib.pyplot as plt
import rk45

def oscillation(theta, t, F, m, L, k, g, omega, eta):
    
    #parameters:
    alpha = (k/((m * L) * np.sqrt(g * L)))
    beta = F / (m * g)\
    
    
        
    return
    
print(rk45.rk45(theta, t, F, m, L, k,))

