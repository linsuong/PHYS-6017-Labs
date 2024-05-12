import numpy as np
import scipy
import scipy.fft as fft
import scipy.signal.windows as windows
import os
import csv
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import base
from base import correlation_coefficient
from base import merge_dates
from base import plot_scatter
from base import cross_correlation

current_directory = os.path.dirname(__file__)

save_directory = os.path.join(current_directory, 'WrittenReport', 'plots')
                              
dataPath = os.path.join(current_directory, "data")

btc = pd.read_csv(os.path.join(dataPath, "Bitstamp_BTCUSD_d.csv"), usecols= ["date", "close"], skiprows= 1)

eth = pd.read_csv(os.path.join(dataPath, "Bitstamp_ETHUSD_d.csv"), usecols= ["date", "close"], skiprows= 1)

ltc = pd.read_csv(os.path.join(dataPath, "Bitstamp_LTCUSD_d.csv"), usecols= ["date", "close"], skiprows= 1)

usdt = pd.read_csv(os.path.join(dataPath, "Bitstamp_USDTUSD_d.csv"), usecols= ["date", "close"], skiprows= 1)

sol = pd.read_csv(os.path.join(dataPath, "Bitfinex_SOLUSD_d.csv"), usecols = ["date", "close"], skiprows= 1)
##################################################################################################

############Pearson Correlation Coefficient####################

################################################################################################################################################################
btc = base.extract_data(btc, 2021, 2023)
print(btc)
BtcEthMatched = merge_dates(btc, eth)
BtcLtcMatched = merge_dates(btc, ltc)
BtcUsdtMatched = merge_dates(btc, usdt)
BtcSolMatched = merge_dates(btc, sol)

btcEthPrice = BtcEthMatched['close_x']
ethPrice = BtcEthMatched['close_y'] 

btcLtcPrice = BtcLtcMatched['close_x']
ltcPrice = BtcLtcMatched['close_y'] 

btcUsdtPrice = BtcUsdtMatched['close_x']
usdtPrice = BtcUsdtMatched['close_y'] 

btcSolPrice = BtcSolMatched['close_x']
solPrice = BtcSolMatched['close_y']

plot_scatter(btcEthPrice, ethPrice, ['BTC', 'ETH'], save_loc= os.path.join(save_directory, 'btcEth'))
plot_scatter(btcSolPrice, solPrice, ['BTC', 'SOL'], save_loc= os.path.join(save_directory, 'btcSol'))
plot_scatter(btcLtcPrice, ltcPrice, ['BTC', 'LTC'], save_loc= os.path.join(save_directory, 'btcLtc'))
plot_scatter(btcUsdtPrice, usdtPrice, ['BTC', 'USDT'], save_loc= os.path.join(save_directory, 'btcUsdt'))
             
lagBtcEth, correlationBtcEth, sdLagBtcEth, sdCorrelationBtcEth, lagBtcEthVal = cross_correlation(btcEthPrice, ethPrice, n_times= 1000)
lagBtcSol, correlationBtcSol, sdLagBtcSol, sdCorrelationBtcSol, lagBtcSolVal = cross_correlation(btcSolPrice, solPrice, n_times= 1000)
lagBtcUsdt, correlationBtcUsdt, sdLagBtcUsdt, sdCorrelationBtcUsdt, lagBtcUsdtVal = cross_correlation(btcUsdtPrice, usdtPrice, n_times= 1000)
lagBtcLtc, correlationBtcLtc, sdLagBtcLtc, sdCorrelationBtcLtc, lagBtcLtcVal = cross_correlation(btcLtcPrice, ltcPrice, n_times= 1000)

#plt.text(200, 0.98, f"$\\sigma$ on time lag = {sdLagBtcEth}")

plt.axvline(x= 0, color= 'black', linestyle= '--')

plt.errorbar(lagBtcEth, correlationBtcEth, color= 'purple', xerr= sdLagBtcEth, yerr= sdCorrelationBtcEth, 
             ecolor= 'black', elinewidth= 1, capsize= 2, errorevery= 3, label = 'BTC ETH')

plt.errorbar(lagBtcSol, correlationBtcSol, color= 'red', xerr= sdLagBtcSol, yerr= sdCorrelationBtcSol, 
             ecolor= 'black', elinewidth= 1, capsize= 2, errorevery= 3, label = 'BTC SOL')

plt.errorbar(lagBtcUsdt, correlationBtcUsdt, color= 'black', xerr= sdLagBtcUsdt, yerr= sdCorrelationBtcUsdt, 
             ecolor= 'black', elinewidth= 1, capsize= 2, errorevery= 3, label = 'BTC USDT')

plt.errorbar(lagBtcLtc, correlationBtcLtc, color= 'green', xerr= sdLagBtcLtc, yerr= sdCorrelationBtcLtc, 
             ecolor= 'black', elinewidth= 1, capsize= 2, errorevery= 3, label = 'BTC LTC')

#plt.plot(lagBtcEth, correlationBtcEth, linewidth = 3, color = 'red')
#plt.plot(lagBtcLtc, correlationBtcLtc, label = 'btc ltc')
#plt.plot(lagBtcUsdt, correlationBtcUsdt, label = 'btc usdt')
plt.grid()
plt.xlim(-20, 20)
plt.ylim(0.94, 1.02)
plt.title('Cross correlation against time lag between cryptocurrency pairs')
plt.ylabel('Cross Correlation')
plt.xlabel('Time Lag (days)')
plt.legend()
plt.savefig(os.path.join(current_directory, 'WrittenReport', 'plots', 'cross_correlation_main.png'))
plt.clf()

print('eth', correlation_coefficient(btcEthPrice, ethPrice))
print('sol', correlation_coefficient(btcSolPrice, solPrice))
print('ltc', correlation_coefficient(btcLtcPrice, ltcPrice))
print('usdt', correlation_coefficient(btcUsdtPrice, usdtPrice))

print(lagBtcEthVal, '\pm', sdLagBtcEth)
print(lagBtcLtcVal,'\pm', sdLagBtcLtc)
print(lagBtcSolVal,'\pm', sdLagBtcSol)
print(lagBtcUsdtVal,'\pm', sdLagBtcUsdt)