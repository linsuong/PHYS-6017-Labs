import numpy as np
from matplotlib import pyplot as plt
import scipy.fft as fft
import scipy.signal.windows as windows
import os

dataPath = r"C:\Repositories\PHYS-6017-Labs\Projects\TimeSeriesAnalysis\data"
#generate sample data:
time = np.linspace(0, 10, 100, endpoint = False)
size = np.size(time)

y = np.sin(2 * np.pi * time) #f = 10

#fft of data
fft_y = fft.fft(y)

plt.plot(fft_y)
plt.show()



#correlation -> np.correlate(array1, array2, mode = "full")
#convolution -> np.convolve(array1, array2)
#binning -> np.digitize(array, bins)
#sampling frequency > 2 * nyquist freq == 1/2(N/T)


