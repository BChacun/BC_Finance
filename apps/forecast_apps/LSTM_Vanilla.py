from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash import dash_table
from app import app
from sklearn.metrics import *
from math import sqrt
from datetime import time
from numpy import array
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

import time
import yfinance as yf
import pandas as pd
import math
import pathlib
import numpy as np



# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()

pd.options.plotting.backend = "plotly"


def separate_data(data, frac):
    ind = math.floor(frac * len(data))
    train = data[:ind]
    test = data[ind:]
    return train, test


def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		
		end_ix = i + n_steps
		
		if end_ix > len(sequence)-1:
			break
		
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)



def layout(tick):
    layout = html.Div([html.I("Enter your hyperparameters :"),
        html.Br(),
        dcc.Input(id="l_lstm", value=0.010, step=0.002, type="number", placeholder="Learning Rate", debounce=True),
        dcc.Input(id="ep_lstm", type="number",value=100, step=10, placeholder="Number of Episode", debounce=True, style={'margin-left' : '1vh',}),
        html.Button('Run', id='btn_run_lstm', n_clicks=0, style={'margin-left' : '1vh',}),
        dcc.Loading(
            id="model-forecast-lstm",
            type="default",
            children=html.Div(id="loading-output-1")
        ),
        
        ])
    return layout

def layout_final(tick, l, ep):
    pd.options.plotting.backend = "plotly"
    
    
    data = yf.Ticker(tick).history(period='5y')
    df = data["Open"].to_frame()
    df.rename(columns={'Open': 'price',  }, inplace=True)
    
    
    
    
    # Split dataset
    res = separate_data(df, 0.8)
    train, test = res[0], res[1]
    len_test = len(test)
    
    start = time.time()

    # define input sequence
    raw_seq = train.values.tolist()
    # choose a number of time steps
    n_steps = 3
    # split into samples
    X, y = split_sequence(raw_seq, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    opt = keras.optimizers.Adam(learning_rate=l)
    model.compile(optimizer=opt, loss='mse')
    # fit model
    model.fit(X, y, epochs=ep, verbose=0)
    
    test_data = train[:-n_steps].values.tolist() + test.values.tolist()
    
    pred = []
    
    for i in range(len_test):
        
        x_input = array(test_data[i:i+n_steps])
        x_input = x_input.reshape((1, n_steps, n_features))
        yhat = model.predict(x_input, verbose=0)
        
        pred.append(yhat.tolist()[0][0])
    
    # Make predictions
    predictions = pd.Series(pred, index=test.index)
    end = time.time()
    duree = end - start
    #mean_error, mae, rmse, mape = indicateurs_stats(test, predictions)
    
    # Create results
    results = df
    results["Prediction"] = predictions
        
    
    
    
    
    

    
    mae = mean_absolute_error(test, predictions)
    rmse = sqrt(mean_squared_error(test, predictions))
    mape = mean_absolute_percentage_error(test, predictions)
    
    
    
    
    
    

    fig = results.plot(
        title="LSTM Vanilla Model", template="simple_white",
                  labels=dict(index="Date", value="price", variable="legend")
        )



  
    

    layout = html.Div([
       
       
    html.P(str(tick) + ' Data : '),
     dash_table.DataTable(
    data=results.to_dict('records'),
    columns=[{'id': c, 'name': c} for c in df.columns],
    page_size=10,

    ),
    dcc.Graph(figure=fig),
    
    
    
    

    html.Div([
        
        
        html.P('MAE : ' + str(mae)),
        html.P('RMSE : ' + str(rmse)),
        html.P('MAPE : ' + str(mape)),
        html.P('Time : ' + str(duree) + ' s'),
        
        ])
    ])
    
    

    return layout



@app.callback(Output('model-forecast-lstm', 'children'),
    
    [Input('btn_run_lstm','n_clicks')],
    
    State('l_lstm','value'),
    State('ep_lstm','value'),
    State('data_forecast', 'value'),
   )


def update_graph( n_clicks, l, ep, data_value):
    if n_clicks:
        return layout_final(data_value, l, ep)
