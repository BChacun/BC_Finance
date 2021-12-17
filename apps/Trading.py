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
from apps.trading_apps import SMA


# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()



df = yf.Ticker("NVDA").history(period='5y')



layout = html.Div([
    html.H1('Algorithmic Trading'),
    html.P('The data used comes from yahoo finance '),
    html.Div([html.P('Data :'),
        dcc.Dropdown(id = 'data',
        options=[
            {'label': 'None', 'value': 'NONE'},
            {'label': 'NVIDIA', 'value': 'NVDA'},
            
        ],
        value='NONE'
    ),
     
     html.P('Model :'),
     
      dcc.Dropdown(id = 'model',
        options=[
            {'label': 'None', 'value': 'NONE'},
            {'label': 'SMA', 'value': 'SMA'},
            
        ],
        value='NONE'
    ),
      
      html.Button('Run', id='btn', n_clicks=0),
      
      html.Div(id="model-content"),

        
    ])
])















             
@app.callback(Output('model-content', 'children'),
    
    [Input('btn','n_clicks')],
    State('page-content','children'),
    State('model', 'value'),
    State('data', 'value'),
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
    
    if value == 'SMA':
        return SMA.layout(data_value)
    else :
        return [
            html.H1("No model selected", className="text-danger"),
            html.Hr(),
            html.P(f"Please choose a model and stock"),
        ]
