from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash import dash_table
import pathlib
from app import app

import pandas as pd
import numpy as np

import yfinance as yf


# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()



df = yf.Ticker("NVDA").history(period='5y')



layout = html.Div([
    html.H1('Algorithmic Trading'),
    html.P('The data used comes from yahoo finance '),
    html.Div([
        
    html.P('NVIDIA Data : '),
     dash_table.DataTable(
    data=df.to_dict('records'),
    columns=[{'id': c, 'name': c} for c in df.columns],
    page_size=10,

    )

        
    ], style = {"height":'20'})
])
             

