from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash import dash_table
import pathlib
from app import app
from datetime import time
import pandas as pd
import numpy as np
import time
import yfinance as yf


import plotly.express as px

import math
from sklearn.metrics import *
from math import sqrt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()

pd.options.plotting.backend = "plotly"


def separate_data(data, frac):
    ind = math.floor(frac * len(data))
    train = data[:ind]
    test = data[ind:]
    return train, test


def indicateurs_stats(test, predictions):
    mean_error = predictions.mean() - test.mean()
    mae = mean_absolute_error(test, predictions)
    rmse = sqrt(mean_squared_error(test, predictions))
    mape = mean_absolute_percentage_error(test, predictions)
    return mean_error, mae, rmse, mape



def layout(tick):
    
    
    data = yf.Ticker(tick).history(period='5y')
    df = data["Open"].to_frame()
    df.rename(columns={'Open': 'price',  }, inplace=True)
    
    
    
    
    # Split dataset
    res = separate_data(df, 0.8)
    train, test = res[0], res[1]

    # Train model
    start = time.time()
    model = SimpleExpSmoothing(train.values, initialization_method="estimated")
    model_fit = model.fit(optimized=True)

    


    # Make predictions
    prediction = model_fit.forecast(len(test))

    predictions = pd.Series(prediction, index=test.index)
    
    
    end = time.time()
    duree = end-start
    
    
    mean_error, mae, rmse, mape = indicateurs_stats(test, predictions)
    
    

    # Create results
    results = df
    #results = results.to_frame()
    results["Prediction"] = predictions
    
    
    
    
    
    
    
    
    
    
    
    
    mae = mean_absolute_error(test, predictions)
    rmse = sqrt(mean_squared_error(test, predictions))
    mape = mean_absolute_percentage_error(test, predictions)
    
    
    
    
    
    
    #df.rename(columns={'price': 'open_price',  }, inplace=True)

   
    
    fig = results.plot(
        title="SES Model", template="simple_white",
                  labels=dict(index="Date", value="price variation", variable="legend")
        )

    

    layout = html.Div([
        
    html.P(str(tick) + ' Data : '),
     dash_table.DataTable(
    data=df.to_dict('records'),
    columns=[{'id': c, 'name': c} for c in df.columns],
    page_size=10,

    ),
    html.P("α = " + str(model.params['smoothing_level'])),
     
    dcc.Graph(figure=fig),
    
    html.Div([
        
        
        html.P('MAE : ' + str(mae)),
        html.P('RMSE : ' + str(rmse)),
        html.P('MAPE : ' + str(mape)),
        
        ])
     
  ])

    return layout




