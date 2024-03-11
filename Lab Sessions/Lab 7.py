#Monte Carlo Simulations
import numpy as np
import random
from matplotlib import pyplot as plt

def func_1(x):
    x = x[0]
    
    return np.abs((np.sin((13/3) * np.pi * x) ** 3 - 2 * np.cos(((2 * np.cos(x/np.pi)) ** 3)) ** 2))

def func_2(x, y):
    x = x[0] 
    y = x[1]
    
    return x ** 2 + 2 * (y ** 2)

def func_3(x, y):
    x = x[0] 
    y = x[1]
    
    return (x ** 2) * (y ** 2)

def func_4(x, y, z):
    x = x[0] 
    y = x[1]
    z = x[2]
    
    np.exp(x * y * z)
    
iteration_numbers = [100, 1000, 10000, 100 000]

#Q1 range = 2.5 to 7.5
a1 = 2.5
b1 = 7.5

def sampler():
    while True:
        yield.random.uniform(a1, b1)
        
for n in iteration_numbers:    
    result_1, error_1 = mcint.integrate(func_1, sampler(), measure = 1.0, n)

for k in iteration_numbers:


#Q2 range x: 0 to 4, y: 0 to 1
a2 = 0
b2 = 4
c2 = 0
d2 = 1

#Q3 range  x: 0 to 4,  y: 0 to 1
a3 = 0
b3 = 4
c3 = 0
d3 = 1

#Q4 range x: 0 to 1, y: 0 to 1, z: 0 to 1
a4 = 0
b4 = 1
c4 = 0
d4 = 1
e4 = 0
f4 = 1

