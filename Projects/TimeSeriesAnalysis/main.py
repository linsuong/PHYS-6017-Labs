import numpy as np
import scipy
import scipy.fft as fft
import scipy.signal.windows as windows
import os
import csv
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from base import correlation_coefficient
from base import merge_dates
from base import plot_scatter
from base import cross_correlation

current_directory = os.path.dirname(__file__)

dataPath = os.path.join(current_directory, "data")

btc = pd.read_csv(os.path.join(dataPath, "Bitstamp_BTCUSD_d.csv"), usecols= ["date", "close"], skiprows= 1)

eth = pd.read_csv(os.path.join(dataPath, "Bitstamp_ETHUSD_d.csv"), usecols= ["date", "close"], skiprows= 1)

#doge = pd.read_csv(os.path.join(dataPath, "Bitstamp_DOGEUSD_d.csv"), usecols= ["date", "close"], skiprows= 1)

ltc = pd.read_csv(os.path.join(dataPath, "Bitstamp_LTCUSD_d.csv"), usecols= ["date", "close"], skiprows= 1)

usdt = pd.read_csv(os.path.join(dataPath,  "Bitstamp_USDTUSD_d.csv"), usecols= ["date", "close"], skiprows= 1)

btcDate = btc["date"]
btcPrice = btc["close"]

btcPrice = np.array(btcPrice)
btcDate = np.array(btcDate)

ethDate = eth["date"]
ethPrice = eth["close"]

ethPrice = np.array(ethPrice)
ethDate = np.array(ethDate)

'''
dogeDate = doge["date"]
dogePrice = doge["close"]

dogePrice = np.array(dogePrice)
dogeDate = np.array(dogeDate)
'''

ltcDate = ltc["date"]
ltcPrice = ltc["close"]

ltcPrice = np.array(ltcPrice)
ltcDate = np.array(ltcDate)

usdtDate = usdt["date"]
usdtPrice = usdt["close"]

usdtPrice = np.array(usdtPrice)
usdtDate = np.array(usdtDate)

##################################################################################################

############Pearson Correlation Coefficient####################

################################################################################################################################################################

correlationBtcEth, lagBtcEth = cross_correlation(btcPrice, ethPrice)
correlationBtcLtc, lagBtcLtc = cross_correlation(btcPrice, ltcPrice)
correlationBtcUsdt, lagBtcUsdt = cross_correlation(btcPrice, usdtPrice)

plt.axvline(x= 0, color= 'black', linestyle= '--')
plt.plot(lagBtcEth, correlationBtcEth, label = 'btc eth')
plt.plot(lagBtcLtc, correlationBtcLtc, label = 'btc ltc')
plt.plot(lagBtcUsdt, correlationBtcUsdt, label = 'btc usdt')
plt.title('cross correlation between cryptocurrency pairs')
plt.ylabel('cross correlation')
plt.xlabel('time lag (days)')
plt.legend()
plt.show()