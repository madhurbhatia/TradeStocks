# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 01:08:32 2019

@author: Madhur
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import joblib
from joblib import Parallel, delayed
import multiprocessing

num_cores = multiprocessing.cpu_count()

data = joblib.load(r'Processed Data.pkl')

names = ["Nearest Neighbors", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA", "XGBoost"]


model_features = [ 'Closing Price 1 Day Offset',
 'Highest Price 1 Day Offset',
 'Lowest Price 1 Day Offset',
 'Opening Price 1 Day Offset',
 'Volume 1 Day Offset',
 'Delta Price 1 Day Offset',
 'Daily Spread Price 1 Day Offset',
 'Change 1 Day Offset',
 'Closing Price 2 Day Offset',
 'Highest Price 2 Day Offset',
 'Lowest Price 2 Day Offset',
 'Opening Price 2 Day Offset',
 'Volume 2 Day Offset',
 'Delta Price 2 Day Offset',
 'Daily Spread Price 2 Day Offset',
 'Change 2 Day Offset',
 'Closing Price 3 Day Offset',
 'Highest Price 3 Day Offset',
 'Lowest Price 3 Day Offset',
 'Opening Price 3 Day Offset',
 'Volume 3 Day Offset',
 'Delta Price 3 Day Offset',
 'Daily Spread Price 3 Day Offset',
 'Change 3 Day Offset',
 'Prev Close Ratio',
 'Volume Adi',
 'Volume Obv',
 'Volume Cmf',
 'Volume Fi',
 'Momentum Mfi',
 'Volume Em',
 'Volume Sma Em',
 'Volume Vpt',
 'Volume Nvi',
 'Volume Vwap',
 'Volatility Atr',
 'Volatility Bbm',
 'Volatility Bbh',
 'Volatility Bbl',
 'Volatility Bbw',
 'Volatility Bbp',
 'Volatility Bbhi',
 'Volatility Bbli',
 'Volatility Kcc',
 'Volatility Kch',
 'Volatility Kcl',
 'Volatility Kcw',
 'Volatility Kcp',
 'Volatility Kchi',
 'Volatility Kcli',
 'Volatility Dcl',
 'Volatility Dch',
 'Trend Macd',
 'Trend Macd Signal',
 'Trend Macd Diff',
 'Trend Sma Fast',
 'Trend Sma Slow',
 'Trend Ema Fast',
 'Trend Ema Slow',
 'Trend Adx',
 'Trend Adx Pos',
 'Trend Adx Neg',
 'Trend Vortex Ind Pos',
 'Trend Vortex Ind Neg',
 'Trend Vortex Ind Diff',
 'Trend Trix',
 'Trend Mass Index',
 'Trend Cci',
 'Trend Dpo',
 'Trend Kst',
 'Trend Kst Sig',
 'Trend Kst Diff',
 'Trend Ichimoku Conv',
 'Trend Ichimoku Base',
 'Trend Ichimoku A',
 'Trend Ichimoku B',
 'Trend Visual Ichimoku A',
 'Trend Visual Ichimoku B',
 'Trend Aroon Up',
 'Trend Aroon Down',
 'Trend Aroon Ind',
 'Trend Psar Up Indicator',
 'Trend Psar Down Indicator',
 'Momentum Rsi',
 'Momentum Tsi',
 'Momentum Uo',
 'Momentum Stoch',
 'Momentum Stoch Signal',
 'Momentum Wr',
 'Momentum Ao',
 'Momentum Kama',
 'Momentum Roc',
 'Others Dr',
 'Others Dlr',
 'Closing Price 3 Day Average',
 'Highest Price 3 Day Average',
 'Lowest Price 3 Day Average',
 'Opening Price 3 Day Average',
 'Volume 3 Day Average',
 'Delta Price 3 Day Average',
 'Daily Spread Price 3 Day Average',
 'Change 3 Day Average',
 'Closing Price 10 Day Average',
 'Highest Price 10 Day Average',
 'Lowest Price 10 Day Average',
 'Opening Price 10 Day Average',
 'Volume 10 Day Average',
 'Delta Price 10 Day Average',
 'Daily Spread Price 10 Day Average',
 'Change 10 Day Average',
 'Closing Price 20 Day Average',
 'Highest Price 20 Day Average',
 'Lowest Price 20 Day Average',
 'Opening Price 20 Day Average',
 'Volume 20 Day Average',
 'Delta Price 20 Day Average',
 'Daily Spread Price 20 Day Average',
 'Change 20 Day Average',
 'Closing Price 50 Day Average',
 'Highest Price 50 Day Average',
 'Lowest Price 50 Day Average',
 'Opening Price 50 Day Average',
 'Volume 50 Day Average',
 'Delta Price 50 Day Average',
 'Daily Spread Price 50 Day Average',
 'Change 50 Day Average',
 'Closing Price 100 Day Average',
 'Highest Price 100 Day Average',
 'Lowest Price 100 Day Average',
 'Opening Price 100 Day Average',
 'Volume 100 Day Average',
 'Delta Price 100 Day Average',
 'Daily Spread Price 100 Day Average',
 'Change 100 Day Average',
 'Closing Price 200 Day Average',
 'Highest Price 200 Day Average',
 'Lowest Price 200 Day Average',
 'Opening Price 200 Day Average',
 'Volume 200 Day Average',
 'Delta Price 200 Day Average',
 'Daily Spread Price 200 Day Average',
 'Change 200 Day Average']

data = data.loc[data['Date'] >= pd.to_datetime('2020-10-01'), ]

result_file_name = 'Results.xlsx'
sell_results = []
buy_results = []
results = pd.read_excel(result_file_name)
results.dropna(inplace = True)

def predict(stock):
    try:
        stock_data = data.loc[data['Stock Index']==stock, ]
        stock_data['Prev Close Ratio'] = stock_data['Closing Price 1 Day Offset']/stock_data['Closing Price 2 Day Offset']
        stock_data['Prediction'] = 0
        stock_results = results.loc[results['Stock Index'] == stock, ]
        for name in names:            
            try:
                model_file_name =  'Models - ' + name + '/' + stock + ' - Model.pkl'
                colname = 'Prediction - ' + name
                model = joblib.load(model_file_name)
                precision = stock_results.loc[stock_results['Classifier'] == name, 'Precision'].values[0]
                stock_data[colname] = model.predict(stock_data[model_features])
                stock_data[colname] = stock_data[colname].map(lambda x: (-1 if x==0 else x))
                stock_data[colname] = precision*stock_data[colname]
                stock_data['Prediction'] = stock_data['Prediction'] + stock_data[colname]
            except:
                continue
        sell_results.append(stock_data.loc[stock_data['Prediction'] > 0, ])
        buy_results.append(stock_data.loc[stock_data['Prediction'] <= 0, ])
    except:
        pass

all_stocks = results['Stock Index'].unique().tolist()

Parallel(n_jobs = num_cores, backend = "threading")(delayed(predict)(stock) for stock in all_stocks)

sell_output = pd.concat(sell_results)
buy_output = pd.concat(buy_results)
sell_output['Model'] = 'Sell'
buy_output['Model'] = 'Buy'
output = pd.concat([sell_output, buy_output])

output = output[['Date', 'Model', 'Stock Index', 'Closing Price 1 Day Offset', 'Volume 20 Day Average', 'Daily Spread Price 20 Day Average', 'Volume 1 Day Offset']]

max_trade_val = 125000
liquidity_ratio = 500

output['Max Volume'] = (output['Volume 20 Day Average']/liquidity_ratio).map(int)
output['Max Investment'] = output['Max Volume']*output['Closing Price 1 Day Offset']
output = output.loc[output['Max Investment'] >= max_trade_val, ]
#output = output.loc[output['Closing Price 1 Day Offset'] > 20, ]

reco_date = output['Date'].max()
output = output.loc[output['Date'] == pd.to_datetime(reco_date), ]
output.rename(columns = {'Closing Price 1 Day Offset': 'Previous Close'}, index = str, inplace = True)
output = output[['Date', 'Model', 'Stock Index', 'Previous Close',  'Daily Spread Price 20 Day Average']]
output['Deviation'] = output['Daily Spread Price 20 Day Average']/output['Previous Close']

def format_deviation(val):
    if val < 0.02:
        return 0.02
    elif val > 0.05:
        return 0.05
    else:
        return val

output['Deviation'] = output['Deviation'].map(lambda x: format_deviation(x))
output['Investment'] = max_trade_val
output['Number of Shares'] = round((output['Investment']/output['Previous Close']), 0)
output = output[['Date', 'Model', 'Stock Index', 'Number of Shares', 'Previous Close', 'Investment', 'Deviation']]
output.to_excel("Today's Recos.xlsx", index = False)

output['Model'].value_counts()


