import numpy as np
import scipy
import scipy.fft as fft
import scipy.signal.windows as windows
import os
import csv
import base
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt


current_directory = os.path.dirname(__file__)

dataPath = os.path.join(current_directory, "data")

btcData = pd.read_csv(os.path.join(dataPath, "Bitstamp_BTCUSD_d.csv"), usecols= ["date", "close"], skiprows= 1)

snpData = pd.read_csv(os.path.join(dataPath, "snp5year.csv"), usecols= ["date", "close"])

goldData = pd.read_csv(os.path.join(dataPath, "gold5year.csv"), usecols= ["date", "close"])

eurData = pd.read_csv(os.path.join(dataPath, "euro-dollar-exchange-rate-historical-chart.csv"), usecols= ['date', 'close'])

btcSearch = pd.read_csv(os.path.join(dataPath, "btcSearchTimeline.csv"), usecols= ["date", "search"])

#in merged_data_202X, close_x is BTC and close_y is S&P500
btcData = base.extract_data(btcData, 2020, 2023)

merged_data = base.merge_dates(btcData, goldData)

btcPrice = merged_data['close_x']
goldPrice = merged_data['close_y']

btcPrice = np.array(btcPrice)

fig, ax = plt.subplots(2, 1, figsize = (9, 13))
lagBtcGold, correlationBtcGold, sdLagBtcGold, sdCorrelationBtcGold, lagvalueBtcGold = base.cross_correlation(btcPrice, goldPrice, n_times= 1000)
spearman_r_gold, pearson_r_gold  = base.correlation_coefficient(btcPrice, goldPrice)
print('gold')
print(spearman_r_gold)
print(pearson_r_gold)

ax[0].grid('True')
ax[0].set_title('Scatter plot between BTC Price and gold spot price from 2020 to 2023')
ax[0].scatter(btcPrice, goldPrice, s = 5, color = 'black')
ax[0].set_xlabel('Price of BTC in USD')
ax[0].set_ylabel('Gold spot price (US dollars per troy ounce)')

ax[1].grid('True')
ax[1].errorbar(lagBtcGold, correlationBtcGold, color= 'purple', xerr= sdLagBtcGold, yerr= sdCorrelationBtcGold,
               ecolor='black', elinewidth=1, capsize=2, errorevery=10)
ax[1].text(-30, 0.375, f"$\\sigma$ on cross-correlation = {np.mean(sdCorrelationBtcGold):.2f},\n$\\sigma$ on time lag = {sdLagBtcGold:.2f} days")
ax[1].axvline(x=0, color='black', linestyle='--')
ax[1].set_xlim(-20, 20)
ax[1].set_ylim(0.985, 1)
ax[1].set_title('Cross correlation against time lag between BTC Price and and gold spot price from 2020 to 2023')
ax[1].set_ylabel('Cross Correlation')
ax[1].set_xlabel('Time Lag (days)')

plt.savefig(os.path.join(current_directory, "WrittenReport", "plots", "btcPriceGoldPrice.png"))
plt.clf()

print(lagvalueBtcGold, '\pm', sdLagBtcGold)
#####################################################################################
#####################################################################################

merged_data = base.merge_dates(btcData, snpData)

btcPrice = merged_data['close_x']
snpPrice = merged_data['close_y']

lagBtcSnp, correlationBtcSnp, sdLagBtcSnp, sdCorrelationBtcSnp, lagValBtcSnp = base.cross_correlation(btcPrice, snpPrice, n_times= 1000)
spearman_r_snp, pearson_r_snp  = base.correlation_coefficient(btcPrice, snpPrice)
print('snp')
print(spearman_r_snp)
print(pearson_r_snp)

fig1, ax1 = plt.subplots(2, 1, figsize = (9, 13))

ax1[0].grid('True')
ax1[0].set_title('Scatter plot between BTC Price and S&P 500 index from 2020 to 2023')
ax1[0].scatter(btcPrice, snpPrice, s = 5, color = 'black')
ax1[0].set_xlabel('Price of BTC in USD')
ax1[0].set_ylabel('S&P500 index price in USD')

ax1[1].grid('True')
ax1[1].errorbar(lagBtcSnp, correlationBtcSnp, color= 'purple', xerr= sdLagBtcSnp, yerr= sdCorrelationBtcSnp,
               ecolor='black', elinewidth=1, capsize=2, errorevery=10)
ax1[1].text(-30, 0.375, f"$\\sigma$ on cross-correlation = {np.mean(sdCorrelationBtcSnp):.2f},\n$\\sigma$ on time lag = {sdLagBtcSnp:.2f} days")
ax1[1].axvline(x=0, color='black', linestyle='--')
ax1[1].set_xlim(-15, 15)
ax1[1].set_ylim(0.985, 1)
ax1[1].set_title('Cross correlation against time lag between BTC Price and S&P 500 index from 2020 to 2023')
ax1[1].set_ylabel('Cross Correlation')
ax1[1].set_xlabel('Time Lag (days)')

plt.savefig(os.path.join(current_directory, "WrittenReport", "plots", "btcPriceSnp.png"))
plt.clf()

print(lagValBtcSnp, '\pm', sdLagBtcSnp)

#####################################################################################
#####################################################################################
merged_data = base.merge_dates(btcData, eurData)

btcPrice = merged_data['close_x']
eurPrice = merged_data['close_y']

lagBtcEur, correlationBtcEur, sdLagBtcEur, sdCorrelationBtcEur, lagValBtcEur = base.cross_correlation(btcPrice, eurPrice, n_times= 1000)
spearman_r_eur, pearson_r_eur  = base.correlation_coefficient(btcPrice, eurPrice)
print('eur')
print(spearman_r_eur)
print(pearson_r_eur)

fig2, ax2 = plt.subplots(2, 1, figsize = (9, 13))

ax2[0].grid('True')
ax2[0].set_title('Scatter plot between BTC Price and S&P 500 index from 2020 to 2023')
ax2[0].scatter(btcPrice, eurPrice, s = 5, color = 'black')
ax2[0].set_xlabel('Price of BTC in USD')
ax2[0].set_ylabel('EUR-USD exchange rate (higher = USD stronger)')

ax2[1].grid('True')
ax2[1].errorbar(lagBtcEur, correlationBtcEur, color= 'purple', xerr= sdLagBtcEur, yerr= sdCorrelationBtcEur,
               ecolor='black', elinewidth=1, capsize=2, errorevery=10)
ax2[1].text(-30, 0.375, f"$\\sigma$ on cross-correlation = {np.mean(sdCorrelationBtcEur):.2f},\n$\\sigma$ on time lag = {sdLagBtcEur:.2f} days")
ax2[1].axvline(x=0, color='black', linestyle='--')
ax2[1].set_xlim(-20, 20)
ax2[1].set_ylim(0.99, 1.002)
ax2[1].set_title('Cross correlation against time lag between BTC Price and EUR-GBP exchange rates from 2020 to 2023')
ax2[1].set_ylabel('Cross Correlation')
ax2[1].set_xlabel('Time Lag (days)')

plt.savefig(os.path.join(current_directory, "WrittenReport", "plots", "btcPriceEur.png"))
plt.clf()

print("eur")
print(lagValBtcEur, '\pm', sdLagBtcEur)

##################################################################################
###################################################################################
merged_data = base.merge_dates(btcData, btcSearch)

btcSearch['search'] = pd.to_numeric(btcSearch['search'], errors='coerce')
btcPrice = merged_data['close']
btcSearch = merged_data['search']

btcPrice = np.array(btcPrice)
btcSearch = np.array(btcSearch)

'''
print(btcSearch)
print(btcPrice)
plt.scatter(btcSearch, btcPrice, s = 0.5, color = 'black')
plt.show()
'''
fig3, ax3 = plt.subplots(2, 1, figsize = (9, 13))
lagBtcSearch, correlationBtcSearch, sdLagBtcSearch, sdCorrelationBtcSearch, lagValBtcSearch = base.cross_correlation(btcPrice, btcSearch, n_times= 1000)
spearman_r, pearson_r  = base.correlation_coefficient(btcPrice, btcSearch)
print('search')
print(spearman_r)
print(pearson_r)

ax3[0].grid('True')
ax3[0].set_title('Scatter plot between BTC Price and \nthe search frequency of the term "Bitcoin" on Google from 2020 to 2023')
ax3[0].scatter(btcPrice, btcSearch, s = 5, color = 'black')
ax3[0].set_ylim(ymin = 0)
ax3[0].set_xlabel('Price of BTC in USD')
ax3[0].set_ylabel('Search interest relative to the highest point (100), a value of 100 is the peak popularity')

ax3[1].grid('True')
ax3[1].errorbar(lagBtcSearch, correlationBtcSearch, color= 'purple', xerr= sdLagBtcSearch, yerr= sdCorrelationBtcSearch,
               ecolor='black', elinewidth=1, capsize=2, errorevery=5)
ax3[1].text(-30, 0.375, f"$\\sigma$ on cross-correlation = {np.mean(sdCorrelationBtcSearch):.2f},\n$\\sigma$ on time lag = {sdLagBtcSearch:.2f} days")
ax3[1].axvline(x=0, color='black', linestyle='--')
ax3[1].set_xlim(-15, 15)
ax3[1].set_ylim(0.75, 1)
ax3[1].set_title('Cross correlation against time lag between BTC Price and \nthe search frequency of the term "Bitcoin" on Google from 2020 to 2023')
ax3[1].set_ylabel('Cross Correlation')
ax3[1].set_xlabel('Time Lag (days)')

plt.savefig(os.path.join(current_directory, "WrittenReport", "plots", "btcPricebtcSearch.png"))

print(lagValBtcSearch, '\pm', sdLagBtcSearch)
