# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 02:09:12 2020

@author: C507170
"""


from kiteconnect import KiteConnect
import pandas as pd
import warnings
import datetime as dt
import math
from joblib import Parallel, delayed
import multiprocessing
warnings.filterwarnings('ignore')
import requests

trade_val = 125000
max_trades = 4

num_cores = multiprocessing.cpu_count()

kite = KiteConnect(api_key="614sjpzfc7a1ieqc")

# kite.login_url()
data = kite.generate_session("QUahdtK1M51qYpRT4HnozP2Wa5SZOOZN", api_secret="ziwfmqsmnmt7vex6zkrp289ng32qh1rz")
kite.set_access_token(data["access_token"])

recos = pd.read_excel(r"Today's Recos.xlsx")

mis_allowed_stocks = pd.read_html(requests.get('https://zerodha.com/margin-calculator/Equity/').content)[0]
mis_allowed_stocks.columns = ['_'.join(col) for col in mis_allowed_stocks.columns]
mis_allowed_stocks.rename(columns = {'Scrip_Scrip': 'Stock Index', 'MIS_Margin %': 'MIS Margin Percentage', 
                                     'MIS_Leverage': 'MIS Leverage', 'CO_Margin %': 'CO Margin Percentage', 
                                     'CO_Leverage': 'CO Leverage'}, index = str, inplace = True)
mis_allowed_stocks.dropna(inplace = True)

recos = recos.loc[recos['Stock Index'].isin(mis_allowed_stocks['Stock Index'].tolist()), ]

recos = recos.loc[recos['Stock Index'].isin(kite_instruments['Stock Index'].tolist()), ]

recos['Trading Symbol'] = 'NSE:' + recos['Stock Index']

buy_stocks = recos.loc[recos['Model'] == 'Buy', 'Trading Symbol'].tolist()
sell_stocks = recos.loc[recos['Model'] == 'Sell', 'Trading Symbol'].tolist()

#Run code from here to obtain latest information
buy_details = kite.quote(buy_stocks)
sell_details = kite.quote(sell_stocks)

buy_data = pd.DataFrame(columns = ['Trading Symbol', 'LTP', 'Open', 'High', 'Low', 'Previous Close', 'Volume', 'Average Ask Price'])
buy_data['Trading Symbol'] = list(buy_details.keys())


def extract_buy_details(symbol):
    buy_data.loc[buy_data['Trading Symbol']==symbol, 'LTP'] = buy_details[symbol]['last_price']
    buy_data.loc[buy_data['Trading Symbol']==symbol, 'Open'] = buy_details[symbol]['ohlc']['open']
    buy_data.loc[buy_data['Trading Symbol']==symbol, 'High'] = buy_details[symbol]['ohlc']['high']
    buy_data.loc[buy_data['Trading Symbol']==symbol, 'Low'] = buy_details[symbol]['ohlc']['low']
    buy_data.loc[buy_data['Trading Symbol']==symbol, 'Previous Close'] = buy_details[symbol]['ohlc']['close']
    buy_data.loc[buy_data['Trading Symbol']==symbol, 'Volume'] = buy_details[symbol]['volume']    
    depth = buy_details[symbol]['depth']['sell']
    p1, p2, p3, p4, p5 = depth[0]['price'], depth[1]['price'], depth[2]['price'], depth[3]['price'], depth[4]['price']
    q1, q2, q3, q4, q5 = depth[0]['quantity'], depth[1]['quantity'], depth[2]['quantity'], depth[3]['quantity'], depth[4]['quantity']
    try:
        avg_ask = (p1*q1 + p2*q2 + p3*q3 + p4*q4 + p5*q5)/(q1+q2+q3+q4+q5)
    except:
        avg_ask = 0
    buy_data.loc[buy_data['Trading Symbol']==symbol, 'Average Ask Price'] = avg_ask  
    
    
    
sell_data = pd.DataFrame(columns = ['Trading Symbol', 'LTP', 'Open', 'High', 'Low', 'Previous Close', 'Volume', 'Average Ask Price'])
sell_data['Trading Symbol'] = list(sell_details.keys())

def extract_sell_details(symbol):
    sell_data.loc[sell_data['Trading Symbol']==symbol, 'LTP'] = sell_details[symbol]['last_price']
    sell_data.loc[sell_data['Trading Symbol']==symbol, 'Open'] = sell_details[symbol]['ohlc']['open']
    sell_data.loc[sell_data['Trading Symbol']==symbol, 'High'] = sell_details[symbol]['ohlc']['high']
    sell_data.loc[sell_data['Trading Symbol']==symbol, 'Low'] = sell_details[symbol]['ohlc']['low']
    sell_data.loc[sell_data['Trading Symbol']==symbol, 'Previous Close'] = sell_details[symbol]['ohlc']['close']
    sell_data.loc[sell_data['Trading Symbol']==symbol, 'Volume'] = sell_details[symbol]['volume']
    depth = sell_details[symbol]['depth']['buy']
    p1, p2, p3, p4, p5 = depth[0]['price'], depth[1]['price'], depth[2]['price'], depth[3]['price'], depth[4]['price']
    q1, q2, q3, q4, q5 = depth[0]['quantity'], depth[1]['quantity'], depth[2]['quantity'], depth[3]['quantity'], depth[4]['quantity']
    try:
        avg_ask = (p1*q1 + p2*q2 + p3*q3 + p4*q4 + p5*q5)/(q1+q2+q3+q4+q5)
    except:
        avg_ask = 0
    sell_data.loc[sell_data['Trading Symbol']==symbol, 'Average Ask Price'] = avg_ask


buy_data['Trading Symbol'].map(lambda x: extract_buy_details(x))
sell_data['Trading Symbol'].map(lambda x: extract_sell_details(x))

buy_data = buy_data.loc[buy_data['Open'] > 0, ]
sell_data = sell_data.loc[sell_data['Open'] > 0, ]


buy_data['Open Price Ratio'] = buy_data['Open']/buy_data['Previous Close']
buy_data.sort_values(by = 'Open Price Ratio', ascending = True, inplace = True)
avg_buy_open_ratio = buy_data['Open Price Ratio'].mean()
#buy_data = buy_data.loc[buy_data['Open Price Ratio'] <= 1.01, ]
buy_data['Quantity'] = (trade_val/buy_data['Open']).map(int)
buy_data['Model'] = 'Buy'
buy_open_ratio = buy_data['Open Price Ratio'].mean()

sell_data['Open Price Ratio'] = sell_data['Open']/sell_data['Previous Close']
sell_data.sort_values(by = 'Open Price Ratio', ascending = False, inplace = True)
avg_sell_open_ratio = sell_data['Open Price Ratio'].mean()
#sell_data = sell_data.loc[sell_data['Open Price Ratio'] >= 0.99, ]
sell_data['Quantity'] = (trade_val/sell_data['Open']).map(int)
sell_data['Model'] = 'Sell'
sell_open_ratio = sell_data['Open Price Ratio'].mean()


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
    buy_trades = max_trades
    sell_trades = max_trades


buy_data = buy_data.head(buy_trades)
sell_data = sell_data.head(sell_trades)

data = pd.concat([buy_data, sell_data])

def print_recos(symbol):
    model = data.loc[data['Trading Symbol'] == symbol, 'Model'].values[0]
    qty = data.loc[data['Trading Symbol'] == symbol, 'Quantity'].values[0]
    price = data.loc[data['Trading Symbol'] == symbol, 'Open'].values[0]
    if model == 'Buy':
        action = "Buy"
    else:
        action = "Sell"
    print(action + ' ' + str(qty) + ' ' + str(symbol.split(':')[1]) + " @ " + str(round(price, 2)))
    return ''

data['Trading Symbol'].map(lambda x: print_recos(x))
print("#########################################################################################")

     

def place_orders(stock):
    symbol = stock.split(':')[1]
    qty = data.loc[data['Trading Symbol'] == stock, 'Quantity'].values[0]
    bet = data.loc[data['Trading Symbol'] == stock, 'Model'].values[0]
#    bid = data.loc[data['Trading Symbol'] == stock, 'Open'].values[0]
    if bet == 'Buy':
#        strike_price = int((bid*0.995)/0.05)*0.05
        order_id = kite.place_order(tradingsymbol = symbol, variety = 'regular',
                                exchange=kite.EXCHANGE_NSE, transaction_type=kite.TRANSACTION_TYPE_BUY,
                                quantity = qty, order_type = 'MARKET', product = 'MIS')
    else:
#        strike_price = int((bid*1.005)/0.05)*0.05
        order_id = kite.place_order(tradingsymbol = symbol, variety = 'regular',
                                exchange=kite.EXCHANGE_NSE, transaction_type=kite.TRANSACTION_TYPE_SELL,
                                quantity = qty, order_type = 'MARKET', product = 'MIS')
    print("Order excuted for " + str(symbol))
    print("Order ID: " + str(order_id))
      

bets = data['Trading Symbol'].tolist()

while(True):
    if dt.datetime.now().time() >= dt.time(9, 15, 2):
        Parallel(n_jobs = num_cores, backend = "threading")(delayed(place_orders)(stock) for stock in bets)
        print("Order Executed!")
        break
    else:
        continue


#while(True):
#    if dt.datetime.now().time() >= dt.time(9, 15, 30):
#        orders = kite.orders()
#        for order in orders:
#            pending_qty = order['pending_quantity']
#            if pending_qty > 0:
#                kite.modify_order(order_id = order['order_id'], quantity = order['pending_quantity'],
#                                  order_type = 'MARKET', variety = order['variety'])        
#                print("Order Modified!")
#            else:
#                continue
#        break
#    else:
#        continue


positions = kite.positions()['day']        
current_positions = pd.DataFrame(columns = ['Trading Symbol', 'Quantity', 'Average Price', 'LTP'])

symbol = []
quantity = []
avg_price = []
ltp = []
pnl = []


for position in positions:
    buy_qty = position['buy_quantity']
    sell_qty = position['sell_quantity']
    product = position['product']
    if (buy_qty == sell_qty)|(product != 'MIS'):
        continue
    else:
        symbol.append(position['tradingsymbol'])
        quantity.append(position['quantity'])
        avg_price.append(position['average_price'])
        ltp.append(position['last_price'])
        pnl.append(position['pnl'])
    

current_positions['Trading Symbol'] = symbol
current_positions['Quantity'] = quantity
current_positions['Average Price'] = avg_price
current_positions['LTP'] = ltp
current_positions['Trading Symbol'] = current_positions['Trading Symbol'].map(lambda x: ('NSE:' + x))
current_positions = pd.merge(current_positions, data[['Trading Symbol', 'Model', 'Open']], on = 'Trading Symbol', how = 'left')

buy_positions = current_positions.loc[current_positions['Model'] == 'Buy', ]
sell_positions = current_positions.loc[current_positions['Model'] == 'Sell', ]

buy_positions['Trigger Price'] = buy_positions['Average Price']*0.95
buy_positions['Limit Price'] = buy_positions['Average Price']*1.04

sell_positions['Trigger Price'] = sell_positions['Average Price']*1.05
sell_positions['Limit Price'] = sell_positions['Average Price']*0.96

buy_positions['Trigger Price'] = buy_positions['Trigger Price'].map(lambda x: math.ceil(x/0.05)*0.05)
buy_positions['Limit Price'] = buy_positions['Limit Price'].map(lambda x: math.floor(x/0.05)*0.05)

sell_positions['Trigger Price'] = sell_positions['Trigger Price'].map(lambda x: math.floor(x/0.05)*0.05)
sell_positions['Limit Price'] = sell_positions['Limit Price'].map(lambda x: math.ceil(x/0.05)*0.05)


def place_buy_exit_orders(stock):
    try:
        symbol = stock.split(':')[1]
        qty = buy_positions.loc[buy_positions['Trading Symbol'] == stock, 'Quantity'].values[0]
        trig_price = buy_positions.loc[buy_positions['Trading Symbol'] == stock, 'Trigger Price'].values[0]
#        limit_price = buy_positions.loc[buy_positions['Trading Symbol'] == stock, 'Limit Price'].values[0]
        stop_order_id = kite.place_order(tradingsymbol = symbol, variety = 'regular',
                                    exchange=kite.EXCHANGE_NSE, transaction_type=kite.TRANSACTION_TYPE_SELL,
                                    quantity = qty, order_type = 'SL-M', product = 'MIS', trigger_price = trig_price)
#        limit_order_id = kite.place_order(tradingsymbol = symbol, variety = 'regular',
#                                    exchange=kite.EXCHANGE_NSE, transaction_type=kite.TRANSACTION_TYPE_SELL,
#                                    quantity = qty, order_type = 'LIMIT', product = 'MIS', price = limit_price)
        print(stop_order_id)
#        print(limit_order_id)
    except:
        pass


def place_sell_exit_orders(stock):
    try:
        symbol = stock.split(':')[1]
        qty = abs(sell_positions.loc[sell_positions['Trading Symbol'] == stock, 'Quantity'].values[0])
        trig_price = sell_positions.loc[sell_positions['Trading Symbol'] == stock, 'Trigger Price'].values[0]
#        limit_price = sell_positions.loc[sell_positions['Trading Symbol'] == stock, 'Limit Price'].values[0]
        
        stop_order_id = kite.place_order(tradingsymbol = symbol, variety = 'regular',
                                    exchange=kite.EXCHANGE_NSE, transaction_type=kite.TRANSACTION_TYPE_BUY,
                                    quantity = qty, order_type = 'SL-M', product = 'MIS', trigger_price = trig_price)
#        limit_order_id = kite.place_order(tradingsymbol = symbol, variety = 'regular',
#                                    exchange=kite.EXCHANGE_NSE, transaction_type=kite.TRANSACTION_TYPE_BUY,
#                                    quantity = qty, order_type = 'LIMIT', product = 'MIS', price = limit_price)
        print(stop_order_id)
#        print(limit_order_id)
    except:
        pass

buy_positions['Trading Symbol'].map(lambda x: place_buy_exit_orders(x))
sell_positions['Trading Symbol'].map(lambda x: place_sell_exit_orders(x))

##############################################################################################


      
#buy_shortlists = buy_data['Trading Symbol'].tolist()
#sell_shortlists = sell_data['Trading Symbol'].tolist()
#
#def place_orders(stock):
#    symbol = stock.split(':')[1]
#    qty = data.loc[data['Trading Symbol'] == stock, 'Quantity'].values[0]
#    bet = data.loc[data['Trading Symbol'] == stock, 'Model'].values[0]
#    price = data.loc[data['Trading Symbol'] == stock, 'Open'].values[0]
##    ltp = data.loc[data['Trading Symbol'] == stock, 'LTP'].values[0]
#    order_type = data.loc[data['Trading Symbol'] == stock, 'Order Type'].values[0]
#    avg_price = data.loc[data['Trading Symbol'] == stock, 'Average Ask Price'].values[0]
#    if order_type == "LIMIT":
#        if bet == 'Buy':
#            bid = price
#            trigger = math.ceil(((bid*0.99)/0.05))*0.05
#            order_id = kite.place_order(tradingsymbol = symbol, variety = 'co',
#                                    exchange=kite.EXCHANGE_NSE, transaction_type=kite.TRANSACTION_TYPE_BUY,
#                                    quantity = qty, order_type = 'LIMIT', product = 'MIS', price = bid, trigger_price = trigger)
#        else:
#            bid = price
#            trigger = math.floor(((bid*1.01)/0.05))*0.05
#            order_id = kite.place_order(tradingsymbol = symbol, variety = 'co',
#                                    exchange=kite.EXCHANGE_NSE, transaction_type=kite.TRANSACTION_TYPE_SELL,
#                                    quantity = qty, order_type = 'LIMIT', product = 'MIS', price = bid, trigger_price = trigger)
#    else:
#        if bet == 'Buy':
#            trigger = math.ceil(((avg_price*0.99)/0.05))*0.05
#            order_id = kite.place_order(tradingsymbol = symbol, variety = 'co',
#                                    exchange=kite.EXCHANGE_NSE, transaction_type=kite.TRANSACTION_TYPE_BUY,
#                                    quantity = qty, order_type = 'MARKET', product = 'MIS', trigger_price = trigger)
#        else:
#            trigger = math.floor(((avg_price*1.01)/0.05))*0.05
#            order_id = kite.place_order(tradingsymbol = symbol, variety = 'co',
#                                    exchange=kite.EXCHANGE_NSE, transaction_type=kite.TRANSACTION_TYPE_SELL,
#                                    quantity = qty, order_type = 'MARKET', product = 'MIS', trigger_price = trigger)
#    print("Order excuted for " + str(symbol))
#    print("Order ID: " + str(order_id))
#      
#
#
#while(True):
#    if dt.datetime.now().time() >= dt.time(9, 15, 5):
#        if buy_trades > 0:
#            buy_details = kite.quote(buy_shortlists)
#            buy_data = pd.DataFrame(columns = ['Trading Symbol', 'LTP', 'Open', 'High', 'Low', 'Previous Close', 'Volume', 'Average Ask Price'])
#            buy_data['Trading Symbol'] = list(buy_details.keys())
#            buy_data['Trading Symbol'].map(lambda x: extract_buy_details(x))
#            buy_data['Open Price Ratio'] = buy_data['Open']/buy_data['Previous Close']
#            buy_data['LTP to Open Ratio'] = buy_data['LTP']/buy_data['Open']
#            buy_data['Order Type'] = buy_data['LTP to Open Ratio'].map(lambda x: ("LIMIT" if x <= 1 else "MARKET"))
#            buy_data['Quantity'] = (trade_val/buy_data['LTP']).map(lambda x: math.ceil(x))
#            buy_data['Model'] = 'Buy'
#        else:
#            pass
#        
#        if sell_trades > 0:
#            sell_details = kite.quote(sell_shortlists)        
#            sell_data = pd.DataFrame(columns = ['Trading Symbol', 'LTP', 'Open', 'High', 'Low', 'Previous Close', 'Volume', 'Average Ask Price'])
#            sell_data['Trading Symbol'] = list(sell_details.keys())
#            sell_data['Trading Symbol'].map(lambda x: extract_sell_details(x))
#            sell_data['Open Price Ratio'] = sell_data['Open']/sell_data['Previous Close']
#            sell_data['LTP to Open Ratio'] = sell_data['LTP']/sell_data['Open']
#            sell_data['Order Type'] = sell_data['LTP to Open Ratio'].map(lambda x: ("LIMIT" if x >= 1 else "MARKET"))
#            sell_data['Quantity'] = (trade_val/sell_data['LTP']).map(lambda x: math.ceil(x))
#            sell_data['Model'] = 'Sell'
#        else:
#            pass
#        
#        data = pd.concat([buy_data, sell_data])
#        bets = data['Trading Symbol'].tolist()
#
#        Parallel(n_jobs = num_cores, backend = "threading")(delayed(place_orders)(stock) for stock in bets)
#        print("Order Executed!")
#        break
#    else:
#        continue
