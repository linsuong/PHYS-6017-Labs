import numpy as np
import scipy
import scipy.fft as fft
import scipy.signal.windows as windows
import os
import csv
import statistics
import pandas as pd
from scipy.signal import find_peaks
from datetime import datetime
from matplotlib import pyplot as plt

def correlation_coefficient(coin_a ,coin_b,  mode = 'both'):
    """calcluates pearson or spearman correlation coefficients

    Args:
        mode (str, optional): picks to calculate pearson or spearman. Defaults to 'both'.
                                choices: 'pearson', 'spearman', 'both's

    Returns:
       spearman, pearson or both correlation coefficients
       returns in the order pearson, spearman for both
    """
    if len(coin_a) != len(coin_b):
        raise Exception("datasets must be of same length. use merge_dates() function first.")
    
    elif len(coin_a) < len(coin_b):
        coin_b = coin_b[:len(coin_a)]
    
    if mode == 'spearman':
        spearman_r = scipy.stats.spearmanr(coin_a, coin_b, alternative = 'less')
        
        return spearman_r
    
    if mode == 'pearson':
        pearson_r = scipy.stats.pearsonr(coin_a, coin_b)
        
        return pearson_r
        
    if mode == 'both':
        spearman_r = scipy.stats.spearmanr(coin_a, coin_b, alternative = 'less')
        pearson_r = scipy.stats.pearsonr(coin_a, coin_b)
        
        return pearson_r, spearman_r
    
def cross_correlation(coin_a, coin_b, n_times):
    total_correlation = [0] * n_times
    total_lag = [0] * n_times
    
    #print(type(total_lag))

    for i in range(n_times):
        rng = np.random.default_rng()
        sample_a = rng.choice(coin_a, size= len(coin_a), replace= True)
        sample_b = rng.choice(coin_b, size= len(coin_b), replace= True)
    
        correlation = scipy.signal.correlate(sample_a, sample_b, mode='full', method='direct')
        lag = scipy.signal.correlation_lags(len(sample_a), len(sample_b), mode='full')
        
        correlation = correlation / max(correlation)
        
        max_corr_index = np.argmax(correlation)
        total_lag[i] = lag[max_corr_index]
        total_correlation[i] = correlation
    
    '''
    print(np.shape(total_correlation))
    total_correlation = np.array(total_correlation)
    total_lag = np.array(total_lag)
    
    std_dev_corr = np.zeros(len(correlation))
    std_dev_lag =  np.zeros(len(lag))
    avg_corr =  np.zeros(len(correlation))
    avg_lag =  np.zeros(len(lag))
    

    for k in range(len(std_dev_corr)):
        std_dev_corr[k] = np.std(total_correlation[:, k])
        std_dev_lag[k] = np.std(total_lag[:, k])
        
        avg_corr[k] = np.average(total_correlation[:, k])
        avg_lag[k] = np.average(total_lag[:, k])
    '''
    
    avg_corr = np.mean(total_correlation, axis=0)
    avg_max_lag = np.mean(total_lag)
    #avg_lag = np.mean(total_lag, axis=0)
    std_dev_corr = np.std(total_correlation, axis=0)
    std_dev_lag = np.std(total_lag)
    
    lag_index = np.argmax(avg_corr)
    lag_value = lag[lag_index]
    print(std_dev_lag)
    
    return lag, avg_corr, std_dev_lag, std_dev_corr, lag_value

def plot_scatter(coin_a, coin_b, names, save_loc, title =''):
    
    plt.grid()
    plt.scatter(coin_a, coin_b, s = 0.7, color = 'black')
    plt.title(f"Scatter plot of price of {names[1]} against {names[0]} {title}")
    plt.xlabel(f"Price of {names[0]} in USD")
    plt.ylabel(f"Price of {names[1]} in USD")
    
    plt.savefig(save_loc)
    plt.clf()
    

def merge_dates(data1, data2, clearNaN = True):
    """ fixes any discrepancies between 2 datasets. matches data2 to data1 based on the dates in data1
    also requires .csv header be named "date" (case sensitive)

    Args:
        data1 (Pandas Series): pandas series data. use pd.read_csv("filepath") to read the .csv file
        
        data2 (Pandas Series): pandas series data where there are lesser dates, 
                                i.e. data 1 is crypto and data 2 is stock prices
                                
        clearNaN (Boolean): if False, keeps NaN values. Defaults to True.
        
    returns
        merged data in the form of pandas series, where x is data1 and y is data2
    """
    
    data1['date'] = pd.to_datetime(data1['date'])
    data2['date'] = pd.to_datetime(data2['date'])
    
    date_range = pd.date_range(start = data2['date'].min(), end = data2['date'].max(), freq = 'D')
    complete_snpData = pd.DataFrame({'date': date_range})
    data2 = pd.merge(complete_snpData, data2, on = 'date', how = 'left')

    merged_data = pd.merge(data1, data2, on = 'date', how = 'inner')
    
        
    if clearNaN is True:
        merged_data.dropna(inplace= True)
        
        return merged_data
    
    else:
        return merged_data

def extract_data(dataset, start, end):
    """
    Reduces data size into the specified year range.
    """
    if start < end:
        dataset['date'] = pd.to_datetime(dataset['date'])

        dataset.set_index('date', inplace=True)

        reducedData = dataset[(dataset.index.year >= start) & (dataset.index.year <= end)]

        reducedData.reset_index(inplace=True)

        return reducedData
    
    else:
        raise ValueError("start year must be before end year!")
