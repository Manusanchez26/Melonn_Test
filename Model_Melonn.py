# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy import stats

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster

from joblib import dump, load
import warnings

warnings.filterwarnings('ignore')


class PredictingPickers:
    
    def __init__(self, start ='2021-03-01', end = '2021-03-30'):
        self.m1 = load('model1.py')
        self.m2 = load('model2.py')
        self.m3 = load('model3.py')
        self.dfOutput = self.predictions(start, end);
        return
# =============================================================================
# Input-> dataframe: format orderId <object>, requestTime <object> and DeliveryOption <object>
# Output-> dataframe transform in format requestTime  <datatime64[ns]> and Delivery <int64>         
# =============================================================================
    def transformData(self, data):

        data['RequestTime'] = pd.to_datetime(data['RequestTime'], format = '%Y/%m/%d %H:%M:%S')
        data = data.drop(['OrderId'], axis = 1)
        data['Delivery'] = data['DeliveryOption'].map({'Mismo dia entre 6:30 pm y 8:30 pm': 1,
                                     'Siguiente dia entre 12:30 pm y 2:30 pm':2,
                                     'Siguiente dia entre las 6:30 pm y 8:30 pm':3},
                                     na_action=None)
        data = data.drop(['DeliveryOption'], axis = 1)
        return data
# =============================================================================
# Input-> dataframe: format requestTime  <datatime64[ns]> and Delivery <int64>
# Output-> dataframe: transform in format requestTime  <datatime64[ns]>, 
# Delivery <int64>, Shift <float64>, DeliveryTime <dateTime64[ns]>
# =============================================================================
    def shiftAssignment(self, data):
        data.loc[(data['Delivery'] == 1), 'Shift'] = 2 
        data.loc[(data['Delivery'] == 2), 'Shift'] = 1  
        data.loc[(data['Delivery'] == 3), 'Shift'] = 1
        
        data.loc[(data['Delivery'] == 1.0), 'DeliveryTime'] = data['RequestTime']
        data.loc[(data['Delivery'] == 2.0), 'DeliveryTime'] = (data['RequestTime'] + pd.Timedelta("1 day"))
        data.loc[(data['Delivery'] == 3.0), 'DeliveryTime'] = (data['RequestTime'] + pd.Timedelta("1 day"))
        
        data['RequestTime'] = pd.to_datetime(data['RequestTime']).dt.date.astype('datetime64')
        data['DeliveryTime'] = pd.to_datetime(data['DeliveryTime']).dt.date.astype('datetime64')
        return data
# =============================================================================
# Input-> start: start date for making the prediction
#         end: end date for making the prediction
# Output-> dataframe: with the number of pickers to be hired per shift 
# =============================================================================

    def predictions(self, start, end):
        newDf = pd.DataFrame(pd.date_range(start= start, end= end), columns = ['Date'])
        newDf['LevelAffect'] = 0
        newDf = newDf.set_index(['Date']) 
        predictions = self.m1.predict(steps=30, exog = newDf['LevelAffect'])
        dfPred = pd.DataFrame(predictions)
        dfPred.index = pd.date_range(start=newDf.index.min(), end=newDf.index.max()) 
        dfPred['Delivery'] = 1
        predictions2 = self.m2.predict(steps=30, exog = newDf['LevelAffect'])
        dfPred2 = pd.DataFrame(predictions2)
        dfPred2.index = pd.date_range(start=newDf.index.min(), end=newDf.index.max()) 
        dfPred2['Delivery'] = 2
        predictions3 = self.m3.predict(steps=30, exog = newDf['LevelAffect'])
        dfPred3 = pd.DataFrame(predictions3)
        dfPred3.index = pd.date_range(start=newDf.index.min(), end=newDf.index.max()) 
        dfPred3['Delivery'] = 3
        dfOutput = pd.concat([dfPred, dfPred2,dfPred3])
        dfOutput = dfOutput.reset_index(level = [0])
        dfOutput = dfOutput.rename(columns ={'pred':'TotalQuantity', 'index':'RequestTime' })
        dfOutput = dfOutput.astype({'TotalQuantity': int})
        dfOutput = self.shiftAssignment(dfOutput)
        dfOutput['TotalPickers'] = round(dfOutput['TotalQuantity']/30) + 1
        dfOutput = dfOutput.set_index(['DeliveryTime']) 
        dfOutput = dfOutput.groupby([pd.Grouper(freq= 'M'), 'Shift']).agg(TotalPickers=('TotalPickers', 'max'))
        return dfOutput