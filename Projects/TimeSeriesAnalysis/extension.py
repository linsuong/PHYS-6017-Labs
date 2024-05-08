import numpy as np
import scipy
import scipy.fft as fft
import scipy.signal.windows as windows
import os
import csv
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from base import cross_correlation
from base import merge_dates
from base import plot_scatter

current_directory = os.path.dirname(__file__)

dataPath = os.path.join(current_directory, "data")

btcData = pd.read_csv(os.path.join(dataPath, "Bitstamp_BTCUSD_d.csv"), usecols= ["date", "close"], skiprows= 1)

snpData2020 = pd.read_csv(os.path.join(dataPath, "2020_INDEX_US_S&P US_SPX.csv"), usecols= ["date", "close"])

snpData2021 = pd.read_csv(os.path.join(dataPath, "2021_INDEX_US_S&P US_SPX.csv"), usecols= ["date", "close"])

#in merged_data_202X, close_x is BTC and close_y is S&P500

data2020 = merge_dates(btcData, snpData2020)
data2021 = merge_dates(btcData, snpData2021)

print(data2020)
print(data2021)

btc2020 = data2020['close_x']
snp2020 = data2020['close_y']

btc2021 = data2021['close_x']
snp2021 = data2021['close_y']

btc2020 = np.array(btc2020)
btc2021 = np.array(btc2021)
snp2020 = np.array(snp2020)
snp2021 = np.array(snp2021)

time_lag_2020, cross_corr_2020 = cross_correlation(btc2020, snp2020)

plt.axvline(x= 0, color= 'black', linestyle= '--')
plt.plot(time_lag_2020, cross_corr_2020)
plt.title('Cross Correlation of BTC and the S&P 500 in the year 2020')
plt.xlabel('Time Lag')
plt.ylabel('Cross Correlation')
plt.grid()
plt.show()
