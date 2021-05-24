# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 01:08:32 2019

@author: Madhur
"""

#Import required packages
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.metrics import classification_report
import pandas as pd
import joblib
import multiprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

result_file_name = 'Results.xlsx'

num_cores = multiprocessing.cpu_count()

#Import dataset with features for all stocks
data = joblib.load('Processed Data.pkl')

all_stocks = data['Stock Index'].unique().tolist()

#Define list of classification algorithms to be tested
classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    xgb.XGBClassifier()]

names = ["Nearest Neighbors", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA", "XGBoost"]


#Define list of features to be used to train the algorithms
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


#Define function to accept change in stock closing price compared to previous day & create label for classification
#Generating label '1' for decline greater than or equal to 1 percent
#Generating label '0' for advance greater than or equal to 1 percent
#Generating label '-1' for all other cases (change in value within 1 percent of last day's close)
def set_model_response(val):
    if val <= 0.99:
        return 1
    elif val >= 1.01:
        return 0
    else:
        return -1
    
all_model_results = []


#Define function to accept stock symbol as input and split data for the stock into train and test sets
#Train all models saved in the list of classifiers created
#Measure Accuracy, Precision, Recall and F-1 Score of each model on test set
#Save model if weighted average precision is more than 55%
def create_model(stock):
    try:
        print("Creating model for Stock: " + str(all_stocks.index(stock) + 1) + " of " + str(len(all_stocks)))
        stock_data = data.loc[data['Stock Index']==stock, ]
        stock_data['Close Price Ratio'] = stock_data['Close']/ stock_data['Closing Price 1 Day Offset']
        stock_data['Prev Close Ratio'] = stock_data['Closing Price 1 Day Offset']/stock_data['Closing Price 2 Day Offset']
        stock_data['Response'] = stock_data['Close Price Ratio'].map(lambda x: set_model_response(x))        
        stock_data = stock_data.loc[stock_data['Response'] != -1, ]
        stock_data.fillna(-1, inplace = True)
        if stock_data.shape[0] <= 5:
            pass
        else:
            X = stock_data.loc[stock_data['Date'] <= pd.to_datetime('2019-12-31'), model_features]
            y = stock_data.loc[stock_data['Date'] <= pd.to_datetime('2019-12-31'), 'Response']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

            #Iterate over classifiers
            for name, model in zip(names, classifiers):
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, pred)
                report = classification_report(y_test, pred, output_dict = True)
                precision = report['weighted avg']['precision']
                recall = report['weighted avg']['recall']
                f1_score = report['weighted avg']['f1-score']
                if precision >= 0.55:
                    print("Model Precision for " + stock + ": " + str(round(precision*100, 2)) + "%")        
                    current_results = pd.DataFrame({'Stock Index': [stock],
                                                    'Classifier': [name],
                                                    'Accuracy': [accuracy],
                                                    'Precision': [precision],
                                                    'Recall': [recall],
                                                    'F-1 Score': [f1_score]})
                    stock_model_filename = 'Models - ' + name + '/' + stock + " - Model.pkl"
                    joblib.dump(model, stock_model_filename)
                    all_model_results.append(current_results)
                else:
                    continue
    except:        
        pass


#Iterate over list of stock symbols to create and save models and publish results
for stock in all_stocks:
    create_model(stock)

#initial_model_results.dropna(inplace = True)
initial_model_results = pd.concat(all_model_results)
print("Total Models Created: " + str(initial_model_results.shape[0]))
print("Average Accuracy: " + str(initial_model_results['Accuracy'].mean()))
print("Average Precision: " + str(initial_model_results['Precision'].mean()))
print("Average Recall: " + str(initial_model_results['Recall'].mean()))
print("Average F-1 Score: " + str(initial_model_results['F-1 Score'].mean()))

#Save results for all classification metrics as an excel workbook
initial_model_results.to_excel(result_file_name, index = False)



