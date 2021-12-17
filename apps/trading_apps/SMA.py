from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash import dash_table
import pathlib
from app import app

import pandas as pd
import numpy as np

import yfinance as yf


import plotly.express as px

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()

pd.options.plotting.backend = "plotly"









def layout(tick):
    
    data = yf.Ticker(tick).history(period='5y')
    
    
    # SMA :
    
    
    df = data["Open"].to_frame()
    df.rename(columns={'Open': 'price',  }, inplace=True)
    
    # SMA hyperparameter :
    
    sma_s = 50
    sma_l = 200
    
    df = df / df.iloc[0]
    
    df["SMA_S"] = df.price.rolling(sma_s).mean()
    df["SMA_L"] = df.price.rolling(sma_l).mean()
    
    
    df["position"] = np.where(df["SMA_S"] > df["SMA_L"], 1, -1)
    
    
    df.rename(columns={'price': 'open_price',  }, inplace=True)
    fig = df.plot(
        title="SMA Model", template="simple_white",
                  labels=dict(index="Date", value="price variation", variable="legend")
        )
    
    layout = html.Div([
        
    html.P('NVIDIA Data : '),
     dash_table.DataTable(
    data=df.to_dict('records'),
    columns=[{'id': c, 'name': c} for c in df.columns],
    page_size=10,

    ),
     
    dcc.Graph(figure=fig),
     
  ])

    return layout




