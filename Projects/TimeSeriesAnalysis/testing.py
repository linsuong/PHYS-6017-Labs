import numpy as np
import scipy
import scipy.signal.windows as windows
import os
import pandas as pd
from matplotlib import pyplot as plt

dataPath = r"C:\Repositories\PHYS-6017-Labs\Projects\TimeSeriesAnalysis\data"

btc = pd.read_csv(os.path.join(dataPath, "Gemini_BTCUSD_d.csv"), usecols= ["date", "close"])

eth = pd.read_csv(os.path.join(dataPath, "Gemini_ETHUSD_d.csv"), usecols= ["date", "close"])

btcPrice = btc["close"].values
ethPrice = eth["close"].values

# Aligning data
min_len = min(len(btcPrice), len(ethPrice))
btcPrice = btcPrice[:min_len]
ethPrice = ethPrice[:min_len]

# Compute cross-correlation
correlation = np.correlate(btcPrice - btcPrice.mean(), ethPrice - ethPrice.mean(), mode='full')
correlation /= np.std(btcPrice) * np.std(ethPrice)

# Generate lags
lags = np.arange(-len(btcPrice) + 1, len(btcPrice))

plt.plot(lags, correlation)
plt.xlabel('Lag')
plt.ylabel('Cross-correlation')
plt.title('Cross-correlation between BTC and ETH prices')
plt.grid(True)
plt.show()
