from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash import dash_table
import pathlib
from app import app

import pandas as pd
import numpy as np

import yfinance as yf


# Models:
from apps.forecast_apps import SES, GRU_Vanilla, LSTM_Vanilla


# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()







layout = html.Div([
    html.H1('Algorithmic Forecasting'),
    html.P('The data used comes from yahoo finance '),
    html.Div([html.P('Data :'),
        dcc.Dropdown(id = 'data_forecast',
        options=[
            {'label': 'None', 'value': 'NONE'},
            {'label': 'NVIDIA', 'value': 'NVDA'},
            
        ],
        value='NONE'
    ),
     
     html.P('Model :'),
     
      dcc.Dropdown(id = 'model_forecast',
        options=[
            {'label': 'None', 'value': 'NONE'},
            {'label': 'GRU_Vanilla', 'value': 'GRU_Vanilla'},
            {'label': 'LSTM_Vanilla', 'value': 'LSTM_Vanilla'},
            {'label': 'SES', 'value': 'SES'},
            
        ],
        value='NONE'
    ),
      
      html.Button('Run', id='btn_run_forecast', n_clicks=0),
      
      html.Div(id="model-forecast-content"),

        
    ])
])















             
@app.callback(Output('model-forecast-content', 'children'),
    
    [Input('btn_run_forecast','n_clicks')],
    State('page-content','children'),
    State('model_forecast', 'value'),
    State('data_forecast', 'value'),
   )
def update_graph( n_clicks,children, value, data_value):
    
    if data_value == "NONE":
        return [
            html.H1("No data selected", className="text-danger"),
            html.Hr(),
            html.P(f"Please choose a stock"),
        ]
    
    if value == "NONE":
        return [
            html.H1("No model selected", className="text-danger"),
            html.Hr(),
            html.P(f"Please choose a model and launch it"),
        ]
    
    if value == 'GRU_Vanilla':
        return GRU_Vanilla.layout(data_value)
    
    if value == 'LSTM_Vanilla':
        return LSTM_Vanilla.layout(data_value)
    
    if value == 'SES':
        return SES.layout(data_value)
    
    
    
    else :
        return [
            html.H1("No model selected", className="text-danger"),
            html.Hr(),
            html.P(f"Please choose a model and stock"),
        ]
