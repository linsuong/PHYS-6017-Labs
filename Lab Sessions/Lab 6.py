import scipy.integrate as integrate
import matplotlib.pyplot as plt
import math
import numpy as np

def function_1(x):
    return (1/(1 + x ** 2))

def function_2(x):
    return 1/x

def function_3(x):
    return np.sin(x **2)

'''
notation:
quadrature - integrate.quad()
fixed-tolerance gaussian - integrate.simps()
trapezium - integrate.quadf()
simpsons - integrate.simps()
'''

x = np.linspace(0, 5)
y1 = function_1(x)

plt.title("$\\frac{1}{1 + x^{2}}$")
plt.plot(x, y1)
plt.text(1.5, 0.05, f"Trapezoid: {integrate.trapezoid(y1, x):.3f}, \nSimpsons: {integrate.simps(y1, x): .3f}, \nGaussian: {integrate.quad(function_1, 0, 5): .3f}")
plt.show()