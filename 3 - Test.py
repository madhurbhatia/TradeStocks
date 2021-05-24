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
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()


#Import dataset with stock prices to simulate trading strategies and view performance
data = joblib.load(r"Processed Data.pkl")
data.drop_duplicates(subset = ['Date', 'Stock Index'], keep = 'first', inplace = True)
all_stocks = data['Stock Index'].unique().tolist()

#Define list of classification models trained for each stock
names = ["Nearest Neighbors", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA", "XGBoost"]


#Define list of model features
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


#Isolate data since beginning of 2020 to test performance
cut_off_date = '2020-01-01'
data = data.loc[data['Date'] >= pd.to_datetime(cut_off_date), ]

#Import dataframe with results of model performance for each stock
result_file_name = 'Results.xlsx'
sell_results = []
buy_results = []
results = pd.read_excel(result_file_name)
results.dropna(inplace = True)

#Define function to accept stock symbol as input and generate predictions for buy/sell for each day 
def predict(stock):
    try:
        stock_data = data.loc[data['Stock Index']==stock, ]
        stock_data['Prev Close Ratio'] = stock_data['Closing Price 1 Day Offset']/stock_data['Closing Price 2 Day Offset']
        stock_data['Prediction'] = 0
        stock_results = results.loc[results['Stock Index'] == stock, ]
        precision_sum = stock_results['Precision'].sum()
        for name in names:
            try:
                model_file_name =  'Models - ' + name + '/' + stock + ' - Model.pkl'
                colname = 'Prediction - ' + name
                model = joblib.load(model_file_name)
                precision = stock_results.loc[stock_results['Classifier'] == name, 'Precision'].values[0]
                stock_data[colname] = [val[1] for val in model.predict_proba(stock_data[model_features])]
                stock_data[colname] = precision*stock_data[colname]
                stock_data['Prediction'] = stock_data['Prediction'] + stock_data[colname]
            except:
                continue
        stock_data['Prediction'] = stock_data['Prediction']/precision_sum
        sell_results.append(stock_data.loc[stock_data['Prediction'] > 0.5, ])
        buy_results.append(stock_data.loc[stock_data['Prediction'] <= 0.5, ])
    except:
        pass


all_stocks = results['Stock Index'].unique().tolist()

Parallel(n_jobs = num_cores, backend = "threading")(delayed(predict)(stock) for stock in all_stocks)

#Save sell predictions for each stock in dataframe 'sell_output'
#Save buy predictions for each stock in dataframe 'buy_output'
sell_output = pd.concat(sell_results)
buy_output = pd.concat(buy_results)

sell_output['Model'] = 'Sell'
buy_output['Model'] = 'Buy'

#Consolidate sell & buy predictions
output = pd.concat([sell_output, buy_output])

#Set maximum investment amount, Liquidity Ratio, Stop Loss, max. number of daily trades
max_trade_val = 150000
liquidity_ratio = 1000
sl = 0.02
max_trades = 4

output = output[['Date', 'Model', 'Stock Index', 'Prediction', 'Closing Price 1 Day Offset', 'Open', 'Low', 'Close', 'High', 'Volume 20 Day Average', 'Daily Spread Price 20 Day Average', 'Prev Close Ratio', 'Volume 1 Day Offset']]
output.drop_duplicates(subset = ['Date', 'Stock Index'], keep = 'first', inplace = True)
output['Open Price Ratio'] = output['Open']/output['Closing Price 1 Day Offset']
output = output.loc[output['Low'] > 20, ]

output.rename(columns = {'Open': 'Day Open'}, index = str, inplace = True)


#Filter stocks for which average traded value for last 20 days is greater than the product of max investment amount & liquidity ratio
output['Max Volume'] = (output['Volume 20 Day Average']/liquidity_ratio)
output['Max Investment'] = output['Max Volume']*output['Closing Price 1 Day Offset']
output = output.loc[output['Max Investment'] >= max_trade_val, ]


#Define function to find daily bets
def find_bets(i):
    try:
        print("Estimating return for day: " + str(i+1))
        current_day = net_days[i]
        current_data = output.loc[output['Date']==pd.to_datetime(current_day), ]
        avg_opn_price_ratio = current_data.groupby(by = 'Model', as_index = False)['Open Price Ratio'].median()
        buy_open_ratio = avg_opn_price_ratio.loc[avg_opn_price_ratio['Model'] == 'Buy', 'Open Price Ratio'].values[0]
        sell_open_ratio = avg_opn_price_ratio.loc[avg_opn_price_ratio['Model'] == 'Sell', 'Open Price Ratio'].values[0]
        if (buy_open_ratio < 1)&(sell_open_ratio > 1):
            buy_trades = max_trades
            sell_trades = max_trades
        elif (buy_open_ratio < 1)&(sell_open_ratio < 1):
            buy_trades = 2*max_trades
            sell_trades = 0
        elif (buy_open_ratio > 1)&(sell_open_ratio > 1):
            buy_trades = 0
            sell_trades = 2*max_trades
        else:
            buy_trades = 0
            sell_trades = 0
        
        buy_recos = current_data.loc[current_data['Model'] == 'Buy', ].groupby(by = 'Stock Index', as_index = False)['Open Price Ratio'].min()
        sell_recos = current_data.loc[current_data['Model'] == 'Sell', ].groupby(by = 'Stock Index', as_index = False)['Open Price Ratio'].min()
        
        buy_recos.sort_values(by = 'Open Price Ratio', ascending = True, inplace = True)
        sell_recos.sort_values(by = 'Open Price Ratio', ascending = False, inplace = True)
        
        buy_recos = buy_recos.head(buy_trades)
        sell_recos = sell_recos.head(sell_trades)
        
        day_recos = pd.concat([buy_recos, sell_recos])
        current_data = current_data.loc[current_data['Stock Index'].isin(day_recos['Stock Index'].tolist()), ]
        all_stock_data = []
        
        def check_status(stock):
            stock_data = current_data.loc[current_data['Stock Index'] == stock, ]
            position = stock_data['Model'].unique()[0]
            if (position == 'Buy')&(buy_trades > 0):
                stock_data['Buy Price'] = stock_data['Day Open'].values[0]
                stop_loss = stock_data['Sell Price'].unique()[0]*(1 - sl)
                day_close = stock_data['Close'].values[0]
                day_low = stock_data['Low'].values[0]
                if day_low <= stop_loss:
                    stock_data['Buy Price'] = stop_loss
                else:
                    stock_data['Buy Price'] = day_close
                all_stock_data.append(stock_data)

            elif (position == 'Sell')&(sell_trades > 0):
                stock_data['Sell Price'] = stock_data['Day Open'].values[0]
                stop_loss = stock_data['Sell Price'].unique()[0]*(1 + sl)
                day_close = stock_data['Close'].values[0]
                day_high = stock_data['High'].values[0]
                if day_high >= stop_loss:
                    stock_data['Buy Price'] = stop_loss
                else:
                    stock_data['Buy Price'] = day_close
                all_stock_data.append(stock_data)
            else:
                pass
        for stock in current_data['Stock Index'].unique().tolist():
            check_status(stock)
        try:
            current_data = pd.concat(all_stock_data)
            day_results = current_data.groupby(['Date', 'Stock Index'], as_index = False)[['Buy Price', 'Sell Price', 'Model']].min()    
            day_results['Date'] = pd.to_datetime(current_day)
            top_predictions.append(day_results)
        except:
            print("Order Not Executed!")
            pass
    except:
        pass

top_predictions = []
output.sort_values(by = 'Date', ascending = True, inplace = True)
net_days = output['Date'].unique().tolist()

#Iterate over days in the calendar year to find bets for each day
for i in range(0, len(net_days)):
    find_bets(i)
    
#Concatenate data for bets
output = pd.concat(top_predictions)


#Measure gross & net profit assuming standard deductions
output['Actual Closing Price Ratio'] = output['Sell Price']/output['Buy Price']
output['Actual Outcome'] = output['Actual Closing Price Ratio'].map(lambda x: int(x > 1))

output['Investment'] = max_trade_val
output['Returns'] = output['Actual Closing Price Ratio']*output['Investment']
output['Gross Profit'] = output['Returns'] - output['Investment']

output['Brokerage'] = min(2*max_trade_val*0.0003, 40)
output['Turnover'] = output['Investment'] + output['Returns']
output['STT'] = 0.00025*output['Returns']
output['Transaction Charges'] = 0.0000325*output['Turnover']
output['Gross Charges'] = output['Brokerage'] + output['Transaction Charges']
output['GST'] = output['Gross Charges']*0.18
output['SEBI Charges'] = (output['Turnover']/10000000)*5
output['Stamp Charges'] = output['Investment']*0.00003
output['Net Charges'] = output['Gross Charges'] + output['STT'] + output['GST'] + output['SEBI Charges'] + output['Stamp Charges']
output['Profit'] = output['Gross Profit'] - output['Net Charges']
output['Returns'] = output['Investment'] + output['Profit']


returns = pd.DataFrame(output.groupby(by = 'Date', as_index = False)[['Returns']].sum())
model_results = output.groupby(by = 'Date', as_index = False).agg({'Investment': 'sum', 'Actual Outcome': 'mean'})
model_results = pd.merge(model_results, returns, on = 'Date', how = 'left')
model_results['Profit'] = model_results['Returns'] - model_results['Investment']
model_results['Week'] = model_results['Date'].map(lambda x: x.week)
model_results = model_results.groupby(by = 'Week', as_index = False).agg({'Investment': 'mean', 'Returns': 'mean', 'Profit': 'sum', 'Actual Outcome': 'mean'})
model_results['Cumulative Profit'] = model_results['Profit'].cumsum()


plt.scatter(model_results['Week'], model_results['Cumulative Profit'])
plt.title('Scatter plot for Cumulative Weekly Profit')
plt.xlabel('Week')
plt.ylabel('Cumulative Profit')
plt.show()

model_results['ROI'] = model_results['Profit']/model_results['Investment']
returns = model_results['Profit'].sum()
print("Model Precision: " + str(round(output['Actual Outcome'].mean()*100, 2)) + '%')
print("Returns: " + str(round(returns)))
avg_weekly_profit = model_results['Profit'].mean()
projected_annual_returns = round(avg_weekly_profit*52)
print("Projected Annual Returns: " + str(projected_annual_returns))
no_of_profit_weeks = model_results.loc[model_results['Profit'] >= 0, ].shape[0]
print("Percentage of Profit Weeks: " + str(round(no_of_profit_weeks/model_results.shape[0]*100, 2)) + '%')
print("Average Weekly Profit: " + str(round(avg_weekly_profit)))
print("Average Weekly Investment: " + str(round(model_results['Investment'].mean())))
print("Average Weekly Return on Investment: " + str(round(model_results['ROI'].mean()*100, 2)) + '%')
print("Average Loss on Loss Weeks: " + str(round(model_results.loc[model_results['Profit'] < 0, 'Profit'].mean())))
print("Average Profit on Profit Weeks: " + str(round(model_results.loc[model_results['Profit'] > 0, 'Profit'].mean())))
print("Median Weekly Profit: " + str(round(model_results['Profit'].quantile(0.5))))
print("Average Monthly Profits: " + str(round(avg_weekly_profit*4)))
print("Risk-Reward Ratio: " + str(-1*round(round(model_results.loc[model_results['Profit'] > 0, 'Profit'].mean())/round(model_results.loc[model_results['Profit'] < 0, 'Profit'].mean()), 2)))


