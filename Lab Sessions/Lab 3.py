import numpy as np
import math
import matplotlib.pyplot as plt

#finding when python fucks up
print('5.8 + 0.2 =', 5.8 + 0.2)

print('5.9 + 0.2 =', 5.9 + 0.2)

print((2/3) ** 0.5 ** 2)

linusisdumb = (100000000000000000000000000000000000000) + (0.0000000000001)

linusisnotdumb = (0.0000000000001) + (100000000000000000000000000000000000000)

print('kkk %.1000f' %linusisdumb)

print('kkk %.1000f' %linusisdumb)

print(0.5 == 0.1 + 0.1 + 0.1 + 0.1 + 0.1)

print((2**(1/4)) ** 4)

print((0.0000000000001 ** 1000000000000000000) + (100000000000000000000000000000000000000) == (100000000000000000000000000000000000000) + (0.0000000000001 ** 1000000000000000000))

print((0.0000000000001 ** 1000000000000000000) + (100000000000000000000000000000000000000))

print((100000000000000000000000000000000000000) + (0.0000000000001 ** 1000000000000000000))

print(((np.e**np.e)**(1/np.e)) == np.e)

print(math.factorial(1000)/math.factorial(999))

print(math.exp(1))

print(np.e)

print(float())

#pink floyd - time
import time
start = time.time()
print("hello")
end = time.time()
delta = end-start
print("time is %.100f" % delta)

###recurrence relation:
def recurrence():
    x = [0] * 50

    for k in range(1, 49):
        x[0] = 1
        x[1] = 1/3
        x[k + 1] = (13/3 * x[k]) - (4/3) * (x[k - 1])

    #print(np.shape(x))
    return x

xprime = [0] * 50
for n in range(1, 50):
    xprime[n] = 3 ** (-n)

xval = recurrence()

difference = [0] * len(xprime)
for k in range(len(xprime)):
    difference[k] = xval[k] - xprime[k]


print(difference == xval)

plotrange = np.arange(1, 51, 1)

fig, ax = plt.subplots(1, 2)
ax[0].plot(plotrange, xval, label ="recurrence relation")
ax[0].plot(plotrange, xprime, label ="numerical solution")
ax[0].legend()

ax[1].plot(plotrange, difference, label = 'difference')
ax[1].legend()

fig.suptitle('Reccurence Relations in Python')


thetas = np.linspace(0, 2 * np.pi, 100)
cosines = np.empty_like(thetas)

for n in range(1, 10):
    yvals = [[] for _ in range(10)]
    print(np.shape(yvals))

    '''
        for j in range(len(thetas)):
        yvals[n][j] = 2 * np.cos((2 * n - 1) * thetas[j]) - np.cos(((2 * n) - 3) * thetas[j])
    '''
print(np.shape(yvals))
