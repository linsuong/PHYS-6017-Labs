import numpy as np
import scipy
import scipy.fft as fft
import scipy.signal.windows as windows
import os
import csv
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt

def correlation_coefficient(coin_a ,coin_b, pair, mode = 'both'):
    if len(coin_a) > len(coin_b):
        coin_a = coin_a[:len(coin_b)]
    
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
    
def cross_correlation(coin_a, coin_b):
    correlation = scipy.signal.correlate(coin_a, coin_b, mode= 'full', method = 'direct')
    lag = scipy.signal.correlation_lags(len(coin_a), len(coin_b), mode= 'full')
    
    correlation = correlation/max(correlation)
    
    return lag, correlation

def plot_scatter(coin_a, coin_b, names):
    
    plt.scatter(coin_a, coin_b, s = 0.7, color = 'black')
    plt.title(f"Scatter plot of price of {names[1]} against {names[0]}")
    plt.xlabel(f"Price of {names[0]} in USD")
    plt.ylabel(f"Price of {names[1]} in USD")
    plt.show()

def merge_dates(data1, data2, clearNaN = True):
    """ fixes any discrepancies between 2 datasets if one has lesser dates than the other.
    only works if len(data1['date] > len(data2['date']). 
    also requires .csv header be named "date" (case sensitive)

    Args:
        data1 (Pandas Series): pandas series data. use pd.read_csv("filepath") to read the .csv file
        
        data2 (Pandas Series): pandas series data where there are lesser dates, 
                                i.e. data 1 is crypto and data 2 is stock prices
                                
        clearNaN (Boolean): if False, keeps NaN values. Defaults to True.
    """
    if len(data1['date']) > len(data2['date']):
        data1['date'] = pd.to_datetime(data1['date'])
        data2['date'] = pd.to_datetime(data2['date'])
        
        date_range = pd.date_range(start = data2['date'].min(), end = data2['date'].max(), freq = 'D')
        complete_snpData = pd.DataFrame({'date': date_range})
        data2 = pd.merge(complete_snpData, data2, on = 'date', how = 'left')

        merged_data = pd.merge(data1, data2, on = 'date', how = 'inner')
        
    else:
        raise ValueError("first data set must have more dates than second data set")
        
    if clearNaN is True:
        merged_data.dropna(inplace= True)
        
        return merged_data
    
    else:
        return merged_data


