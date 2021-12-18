from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash import dash_table
from app import app
from sklearn.metrics import *
from math import sqrt
from datetime import time
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dense

import time
import yfinance as yf
import plotly.express as px
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




def layout(tick):
    layout = html.Div([html.I("Enter your hyperparameters :"),
        html.Br(),
        dcc.Input(id="l_gru", value=0.010, step=0.002, type="number", placeholder="Learning Rate", debounce=True),
        dcc.Input(id="ep_gru", type="number",value=100, step=10, placeholder="Number of Episode", debounce=True, style={'margin-left' : '1vh',}),
        html.Button('Run', id='btn_run_gru', n_clicks=0, style={'margin-left' : '1vh',}),
        html.Div([],id='model-forecast-gru')
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

    # For the model
    shifted_df = df.shift()
    concat_df = [df, shifted_df]
    data = pd.concat(concat_df, axis=1)
    data.fillna(0, inplace=True)
    res = separate_data(data, 0.8)
    train_m, test_m = res[0], res[1]

    # Scaler used
    scaler = MinMaxScaler()

    # Transform data and fit scaler
    train_scaled = scaler.fit_transform(np.array(train_m))
    test_scaled = scaler.transform(np.array(test_m))
    y_train = train_scaled[:, -1]
    X_train = train_scaled[:, 0:-1]
    X_train = X_train.reshape(len(X_train), 1, 1)
    y_test = test_scaled[:, -1]
    X_test = test_scaled[:, 0:-1]

    
    # Model :
    model = Sequential()

    model.add(GRU(75, input_shape=(1, 1)))
    model.add(Dense(2))
    opt = keras.optimizers.Adam(learning_rate=l)  # learning_rate=0.01
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    # model.fit(X_train, y_train, epochs=100, batch_size=20, shuffle=False)

    start = time.time()
    model.fit(X_train, y_train, epochs=ep, batch_size=20, shuffle=False)
  
    X_test = X_test.reshape(len_test, 1, 1)

    y_pred = model.predict(X_test)
    

    predictions_trad = scaler.inverse_transform(y_pred)
    predictions_col = []
    for i in predictions_trad:
        predictions_col.append(i[0])


    # Make predictions
    predictions = pd.Series(predictions_col, index=test.index)
    end = time.time()
    duree = end - start
    #mean_error, mae, rmse, mape = indicateurs_stats(test, predictions)

    # Create results
    results = df
    results["Prediction"] = predictions
    
    
    
    
    
    

    
    
    
    
    
    
    

    fig = results.plot(
        title="GRU Model", template="simple_white",
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

     
    ])
    
    

    return layout



@app.callback(Output('model-forecast-gru', 'children'),
    
    [Input('btn_run_gru','n_clicks')],
    
    State('l_gru','value'),
    State('ep_gru','value'),
    State('data_forecast', 'value'),
   )


def update_graph( n_clicks, l, ep, data_value):
    if n_clicks:
        return layout_final(data_value, l, ep)
