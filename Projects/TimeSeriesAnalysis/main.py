import numpy as np
import scipy.fft as fft
import scipy.signal.windows as windows
import os
import csv
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt

dataPath = r"C:\Repositories\PHYS-6017-Labs\Projects\TimeSeriesAnalysis\data"

btc = pd.read_csv(os.path.join(dataPath, "Gemini_ETHUSD_d.csv"), usecols= ["date", "close"])

eth = pd.read_csv(os.path.join(dataPath, "Gemini_ETHUSD_d.csv"), usecols= ["date", "close"])

doge = pd.read_csv(os.path.join(dataPath, "Gemini_DOGEUSD_d.csv"), usecols= ["date", "close"])

ltc = pd.read_csv(os.path.join(dataPath, "Gemini_LTCUSD_d.csv"), usecols= ["date", "close"])

btcDate = btc["date"]
btcPrice = btc["close"]

#btcDate = btcDate[::-1]
#btcPrice = btcPrice[::-1]

btcPrice = np.array(btcPrice)
btcDate = np.array(btcDate)

ethDate = eth["date"]
ethPrice = eth["close"]

#ethDate = ethDate[::-1]
#ethPrice = ethPrice[::-1]

ethPrice = np.array(ethPrice)
ethDate = np.array(ethDate)

dogeDate = doge["date"]
dogePrice = doge["close"]

#dogeDate = dogeDate[::-1]
#dogePrice = dogePrice[::-1]

dogePrice = np.array(dogePrice)
dogeDate = np.array(dogeDate)

ltcDate = ltc["date"]
ltcPrice = ltc["close"]

#ltcDate = ltcDate[::-1]
#ltcPrice = ltcPrice[::-1]

ltcPrice = np.array(ltcPrice)
ltcDate = np.array(ltcDate)

################################################################################################################################################################

#plt.plot(btcDate, btcPrice)
plt.plot(ethDate, ethPrice)
plt.plot(ltcDate, ltcPrice)
plt.plot(dogeDate, dogePrice)
plt.show()