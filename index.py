import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
import dash
import base64, os

# Connect to main app.py file
from app import app, server

# Connect to your app pages
from apps import CV, Trading

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("MENU", className="display-4"),
        html.Hr(),
        html.P(
            "Naviguez ici", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Home", href="/", active="exact"),
                dbc.NavLink("Mon CV", href="/apps/CV", active="exact"),
                dbc.NavLink("Trading", href="/apps/Trading", active="exact"),
                
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


"""
Home = html.Div(children=[
    html.Img(src=app.get_asset_url('moi.png'), style={'height':'30%', 'width':'30%'})
, html.Div(children=[html.Img(src=app.get_asset_url('imta.png'), style={'height':'10%', 'width':'10%',"margin-left": 10}),
    html.Img(src=app.get_asset_url('imta.png'), style={'height':'10%', 'width':'10%',"margin-left": 10})  
                     ],style={'display': 'inline'})
], style={'display': 'block'})
               
 """
card = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Baptiste Chacun"),
            html.H2("Ingénieur"),
            html.P("Spécialisé en Data Science"),
            html.Img(src=app.get_asset_url('imta.png'), style={'height':'10%', 'width':'10%',"margin-left": 10})
        ]
    )
)

Home = html.Div([
    dbc.Row([
        html.Img(src=app.get_asset_url('moi.png'), style={'height':'30%', 'width':'30%'}), dbc.Col([card])
]),


])                    
                     
                     


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return Home
    elif pathname == "/apps/CV":
        return CV.layout
    elif pathname == "/apps/Trading":
        return Trading.layout

    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

if __name__ == '__main__':
    app.run_server(debug=True)