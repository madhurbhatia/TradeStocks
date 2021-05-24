# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 01:08:32 2019

@author: Madhur
"""

#Import required packages
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import joblib
from joblib import Parallel, delayed
import multiprocessing
import datetime
import re
import os
import ta

num_cores = multiprocessing.cpu_count()

data = joblib.load('All Stocks.pkl')
all_files = os.listdir("C:/Users/C507170/Documents/Stock Data/Latest Data/")
all_data = []
for file in all_files:
    file = "C:/Users/C507170/Documents/Stock Data/Latest Data/" + file 
    latest_data = pd.read_table(file, delimiter = ',', names = ['Stock Index', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    all_data.append(latest_data)
    
latest_data = pd.concat(all_data)
latest_data.dropna(inplace = True)
latest_data['Date'] = latest_data['Date'].map(str)
latest_data = latest_data.loc[latest_data['Date']!=" delimiter = '", ]
latest_data['Date'] = pd.to_datetime(latest_data['Date'], dayfirst = False)
unique_stocks = data['Stock Index'].unique().tolist()
latest_data = latest_data.loc[latest_data['Stock Index'].isin(unique_stocks), ]
data = data[latest_data.columns]
data = pd.concat([data, latest_data])

for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = data[col].map(float)

data = data.loc[data['Open'] > 0, ]
data = data.loc[data['Low'] > 0, ]
data = data.loc[data['Volume'] > 0, ]
data = data.loc[data['High'] > 0, ]
data = data.loc[data['Close'] > 0, ]
data['Delta'] = data['Close'] - data['Open']
data['Daily Spread'] = data['High'] - data['Low']
data['Change'] = data['Close']/data['Open']

all_stocks = data['Stock Index'].unique().tolist()
data.drop_duplicates(subset = ['Date', 'Stock Index'], keep = 'first', inplace = True)

joblib.dump(data, "All Stocks.pkl")

ta_features = ['volume_adi',
 'volume_obv',
 'volume_cmf',
 'volume_fi',
 'momentum_mfi',
 'volume_em',
 'volume_sma_em',
 'volume_vpt',
 'volume_nvi',
 'volume_vwap',
 'volatility_atr',
 'volatility_bbm',
 'volatility_bbh',
 'volatility_bbl',
 'volatility_bbw',
 'volatility_bbp',
 'volatility_bbhi',
 'volatility_bbli',
 'volatility_kcc',
 'volatility_kch',
 'volatility_kcl',
 'volatility_kcw',
 'volatility_kcp',
 'volatility_kchi',
 'volatility_kcli',
 'volatility_dcl',
 'volatility_dch',
 'trend_macd',
 'trend_macd_signal',
 'trend_macd_diff',
 'trend_sma_fast',
 'trend_sma_slow',
 'trend_ema_fast',
 'trend_ema_slow',
 'trend_adx',
 'trend_adx_pos',
 'trend_adx_neg',
 'trend_vortex_ind_pos',
 'trend_vortex_ind_neg',
 'trend_vortex_ind_diff',
 'trend_trix',
 'trend_mass_index',
 'trend_cci',
 'trend_dpo',
 'trend_kst',
 'trend_kst_sig',
 'trend_kst_diff',
 'trend_ichimoku_conv',
 'trend_ichimoku_base',
 'trend_ichimoku_a',
 'trend_ichimoku_b',
 'trend_visual_ichimoku_a',
 'trend_visual_ichimoku_b',
 'trend_aroon_up',
 'trend_aroon_down',
 'trend_aroon_ind',
 'trend_psar_up',
 'trend_psar_down',
 'trend_psar_up_indicator',
 'trend_psar_down_indicator',
 'momentum_rsi',
 'momentum_tsi',
 'momentum_uo',
 'momentum_stoch',
 'momentum_stoch_signal',
 'momentum_wr',
 'momentum_ao',
 'momentum_kama',
 'momentum_roc',
 'others_dr',
 'others_dlr',
 'others_cr',
 ]

def stock_features_creator(stock):
    print("Processing for stock: " + str(all_stocks.index(stock) + 1) + " of " + str(len(all_stocks)))
    stock_data = data.loc[data['Stock Index']==stock, ]
    stock_data.sort_values(by = 'Date', inplace = True)
    stock_data['Closing Price 1 Day Offset'] = stock_data['Close'].shift(periods = 1)
    stock_data['Highest Price 1 Day Offset'] = stock_data['High'].shift(periods = 1)
    stock_data['Lowest Price 1 Day Offset'] = stock_data['Low'].shift(periods = 1)
    stock_data['Opening Price 1 Day Offset'] = stock_data['Open'].shift(periods = 1)
    stock_data['Volume 1 Day Offset'] = stock_data['Volume'].shift(periods = 1)
    stock_data['Delta Price 1 Day Offset'] = stock_data['Delta'].shift(periods = 1)
    stock_data['Daily Spread Price 1 Day Offset'] = stock_data['Daily Spread'].shift(periods = 1)
    stock_data['Change 1 Day Offset'] = stock_data['Change'].shift(periods = 1)
    
    stock_data = stock_data.loc[stock_data['Volume 1 Day Offset'].notnull(), ]
    stock_data.fillna(0, inplace = True)
    stock_data = ta.add_all_ta_features(stock_data, open = 'Opening Price 1 Day Offset', 
                                        high = 'Highest Price 1 Day Offset', 
                                        low = 'Lowest Price 1 Day Offset', 
                                        close = 'Closing Price 1 Day Offset', 
                                        volume = 'Volume 1 Day Offset')
    
    stock_data['Closing Price 2 Day Offset'] = stock_data['Close'].shift(periods = 2)
    stock_data['Highest Price 2 Day Offset'] = stock_data['High'].shift(periods = 2)
    stock_data['Lowest Price 2 Day Offset'] = stock_data['Low'].shift(periods = 2)
    stock_data['Opening Price 2 Day Offset'] = stock_data['Open'].shift(periods = 2)
    stock_data['Volume 2 Day Offset'] = stock_data['Volume'].shift(periods = 2)
    stock_data['Delta Price 2 Day Offset'] = stock_data['Delta'].shift(periods = 2)
    stock_data['Daily Spread Price 2 Day Offset'] = stock_data['Daily Spread'].shift(periods = 2)
    stock_data['Change 2 Day Offset'] = stock_data['Change'].shift(periods = 2)
    
    stock_data['Prev Close Ratio'] = stock_data['Closing Price 1 Day Offset']/stock_data['Closing Price 2 Day Offset']
    
    stock_data['Closing Price 3 Day Offset'] = stock_data['Close'].shift(periods = 3)
    stock_data['Highest Price 3 Day Offset'] = stock_data['High'].shift(periods = 3)
    stock_data['Lowest Price 3 Day Offset'] = stock_data['Low'].shift(periods = 3)
    stock_data['Opening Price 3 Day Offset'] = stock_data['Open'].shift(periods = 3)
    stock_data['Volume 3 Day Offset'] = stock_data['Volume'].shift(periods = 3)
    stock_data['Delta Price 3 Day Offset'] = stock_data['Delta'].shift(periods = 3)
    stock_data['Daily Spread Price 3 Day Offset'] = stock_data['Daily Spread'].shift(periods = 3)
    stock_data['Change 3 Day Offset'] = stock_data['Change'].shift(periods = 3)

    stock_data['Closing Price 3 Day Average'] = stock_data['Closing Price 1 Day Offset'].rolling(window = 3).mean()
    stock_data['Highest Price 3 Day Average'] = stock_data['Highest Price 1 Day Offset'].rolling(window = 3).mean()
    stock_data['Lowest Price 3 Day Average'] = stock_data['Lowest Price 1 Day Offset'].rolling(window = 3).mean()
    stock_data['Opening Price 3 Day Average'] = stock_data['Opening Price 1 Day Offset'].rolling(window = 3).mean()
    stock_data['Volume 3 Day Average'] = stock_data['Volume 1 Day Offset'].rolling(window = 3).mean()
    stock_data['Delta Price 3 Day Average'] = stock_data['Delta Price 1 Day Offset'].rolling(window = 3).mean()
    stock_data['Daily Spread Price 3 Day Average'] = stock_data['Daily Spread Price 1 Day Offset'].rolling(window = 3).mean()
    stock_data['Change 3 Day Average'] = stock_data['Change 1 Day Offset'].rolling(window = 3).mean()
    
    stock_data['Closing Price 10 Day Average'] = stock_data['Closing Price 1 Day Offset'].rolling(window = 10).mean()
    stock_data['Highest Price 10 Day Average'] = stock_data['Highest Price 1 Day Offset'].rolling(window = 10).mean()
    stock_data['Lowest Price 10 Day Average'] = stock_data['Lowest Price 1 Day Offset'].rolling(window = 10).mean()
    stock_data['Opening Price 10 Day Average'] = stock_data['Opening Price 1 Day Offset'].rolling(window = 10).mean()
    stock_data['Volume 10 Day Average'] = stock_data['Volume 1 Day Offset'].rolling(window = 10).mean()
    stock_data['Delta Price 10 Day Average'] = stock_data['Delta Price 1 Day Offset'].rolling(window = 10).mean()
    stock_data['Daily Spread Price 10 Day Average'] = stock_data['Daily Spread Price 1 Day Offset'].rolling(window = 10).mean()
    stock_data['Change 10 Day Average'] = stock_data['Change 1 Day Offset'].rolling(window = 10).mean()
    
    stock_data['Closing Price 20 Day Average'] = stock_data['Closing Price 1 Day Offset'].rolling(window = 20).mean()
    stock_data['Highest Price 20 Day Average'] = stock_data['Highest Price 1 Day Offset'].rolling(window = 20).mean()
    stock_data['Lowest Price 20 Day Average'] = stock_data['Lowest Price 1 Day Offset'].rolling(window = 20).mean()
    stock_data['Opening Price 20 Day Average'] = stock_data['Opening Price 1 Day Offset'].rolling(window = 20).mean()
    stock_data['Volume 20 Day Average'] = stock_data['Volume 1 Day Offset'].rolling(window = 20).mean()
    stock_data['Delta Price 20 Day Average'] = stock_data['Delta Price 1 Day Offset'].rolling(window = 20).mean()
    stock_data['Daily Spread Price 20 Day Average'] = stock_data['Daily Spread Price 1 Day Offset'].rolling(window = 20).mean()
    stock_data['Change 20 Day Average'] = stock_data['Change 1 Day Offset'].rolling(window = 20).mean()
    
    stock_data['Closing Price 50 Day Average'] = stock_data['Closing Price 1 Day Offset'].rolling(window = 50).mean()
    stock_data['Highest Price 50 Day Average'] = stock_data['Highest Price 1 Day Offset'].rolling(window = 50).mean()
    stock_data['Lowest Price 50 Day Average'] = stock_data['Lowest Price 1 Day Offset'].rolling(window = 50).mean()
    stock_data['Opening Price 50 Day Average'] = stock_data['Opening Price 1 Day Offset'].rolling(window = 50).mean()
    stock_data['Volume 50 Day Average'] = stock_data['Volume 1 Day Offset'].rolling(window = 50).mean()
    stock_data['Delta Price 50 Day Average'] = stock_data['Delta Price 1 Day Offset'].rolling(window = 50).mean()
    stock_data['Daily Spread Price 50 Day Average'] = stock_data['Daily Spread Price 1 Day Offset'].rolling(window = 50).mean()
    stock_data['Change 50 Day Average'] = stock_data['Change 1 Day Offset'].rolling(window = 50).mean()
    
    stock_data['Closing Price 100 Day Average'] = stock_data['Closing Price 1 Day Offset'].rolling(window = 100).mean()
    stock_data['Highest Price 100 Day Average'] = stock_data['Highest Price 1 Day Offset'].rolling(window = 100).mean()
    stock_data['Lowest Price 100 Day Average'] = stock_data['Lowest Price 1 Day Offset'].rolling(window = 100).mean()
    stock_data['Opening Price 100 Day Average'] = stock_data['Opening Price 1 Day Offset'].rolling(window = 100).mean()
    stock_data['Volume 100 Day Average'] = stock_data['Volume 1 Day Offset'].rolling(window = 100).mean()
    stock_data['Delta Price 100 Day Average'] = stock_data['Delta Price 1 Day Offset'].rolling(window = 100).mean()
    stock_data['Daily Spread Price 100 Day Average'] = stock_data['Daily Spread Price 1 Day Offset'].rolling(window = 100).mean()
    stock_data['Change 100 Day Average'] = stock_data['Change 1 Day Offset'].rolling(window = 100).mean()
    
    stock_data['Closing Price 200 Day Average'] = stock_data['Closing Price 1 Day Offset'].rolling(window = 200).mean()
    stock_data['Highest Price 200 Day Average'] = stock_data['Highest Price 1 Day Offset'].rolling(window = 200).mean()
    stock_data['Lowest Price 200 Day Average'] = stock_data['Lowest Price 1 Day Offset'].rolling(window = 200).mean()
    stock_data['Opening Price 200 Day Average'] = stock_data['Opening Price 1 Day Offset'].rolling(window = 200).mean()
    stock_data['Volume 200 Day Average'] = stock_data['Volume 1 Day Offset'].rolling(window = 200).mean()
    stock_data['Delta Price 200 Day Average'] = stock_data['Delta Price 1 Day Offset'].rolling(window = 200).mean()
    stock_data['Daily Spread Price 200 Day Average'] = stock_data['Daily Spread Price 1 Day Offset'].rolling(window = 200).mean()
    stock_data['Change 200 Day Average'] = stock_data['Change 1 Day Offset'].rolling(window = 200).mean()
    all_stock_data.append(stock_data)

all_stock_data = []
data.drop_duplicates(subset = ['Date', 'Stock Index'], keep = 'first', inplace = True)
data = data.reset_index(drop = True)
reco_data = pd.DataFrame(columns = data.columns.tolist())
reco_data['Stock Index'] = all_stocks
reco_date = str(latest_data['Date'].max() + datetime.timedelta(1))
reco_data['Date'] = pd.to_datetime(reco_date)
data = pd.concat([data, reco_data])
Parallel(n_jobs = num_cores, backend = "threading")(delayed(stock_features_creator)(stock) for stock in all_stocks)
data = pd.concat(all_stock_data)
data.drop_duplicates(subset = ['Date', 'Stock Index'], keep = 'first', inplace = True)
data.dropna(axis = 1, inplace = True, how = 'all')

rename_cols = {}
for col in ta_features:
    text = re.sub('_', ' ', col)
    text = text.title()
    rename_cols[col] = text
data.rename(columns = rename_cols, index = str, inplace = True)

joblib.dump(data, 'Processed Data.pkl')

