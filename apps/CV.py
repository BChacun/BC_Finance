from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pathlib
from app import app

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()



layout = html.Iframe(src=app.get_asset_url('CV.PDF'),style={"height": "95vh",'position':'absolute', 'width':'75%'})
                   

