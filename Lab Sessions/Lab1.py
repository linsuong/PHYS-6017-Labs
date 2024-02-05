import matplotlib.pyplot as plt
import numpy as np
import math


##Question 1
x = np.linspace(-10, 10, 100)

plt.plot(x, (-x + 1)/2, color = 'red', label = '2x + 2y = 1')
plt.plot(x, 3/2 * x, color = 'blue', label = '3x - 2y = 0')
plt.title('Question 1')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc = 'upper right', frameon = True)
plt.show()

##Question 2

plt.plot(x, (1 - x), color = 'blue', label = 'x + y = 1')
plt.plot(x,  x ** 2 - 5, color = 'green', label = 'y = x^2 - 5')
plt.title('Question 2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc = 'upper right', frameon = True)
plt.show()

##Question 3

x = np.linspace(0, 3, 100)

plt.plot(x, np.sqrt(9 - x ** 2), color = 'blue', label = 'x^2 + y^2 = 9')
plt.plot(x, -np.sqrt(9 - x ** 2), color = 'blue')
plt.plot(x, x ** 2 + 3 * x + 1, color = 'green', label = 'y = x^2 + 3^x + 1')
plt.title('Question 3')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc = 'upper right', frameon = True)
plt.show()

##Question 4

x = np.arange(-3, 20)

log_val = []
for i in x - 1:
    k = math.log2(x[i] + 4)
    log_val.append(k)
    
print(log_val)

plt.plot(x, log_val, color = 'blue', label = 'log2(x + 4) = y')
plt.plot(x, 3 - x, color = 'red', label = 'y = 3-x')
plt.title('Question 4')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc = 'upper right', frameon = True)
plt.show()

##3 - Estimating integrals graphically##
#Question 1
fig, ax = plt.subplots()
x = np.linspace(-np.pi, np.pi, 100)

for i in range 
plt.plot(x, np.cos(x), color = 'blue', label = 'cos(x)')
plt.show()

#Question 2
x = np.linspace(-np.pi, np.pi/2, 100)

plt.plot(x, np.cos(x)**2 - np.sin(x), color = 'green', label ='cos^2(x) - sin(x)')
plt.show()

#TODO: use patches.rectangle to create a rectangle
##https://stackoverflow.com/questions/37435369/how-to-draw-a-rectangle-on-image
##https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html


##Question 3
x = np.linspace(-10, 0, 100)
y = np.e**x - 10 ** x

plt.show(x, y)

##Question 4
x = np.linspace(0, 20, 100)
y = np.ln(x) - np.e**x + 10

plt.plot(x, y)
plt.show()

