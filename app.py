
import dash
import dash_bootstrap_components as dbc


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)

# ,suppress_callback_exceptions=True

server = app.server
