import dash
import dash_html_components as html
import dash_core_components as dcc
from app.callbacks import register_callbacks

app = dash.Dash(__name__, url_base_pathname='/dashboard/')
app.config.suppress_callback_exceptions = True
register_callbacks(app)

# Layout of the app
app.layout = html.Div(
    children=[
        html.Div(
            html.H2(html.A('Active Learning Explorer',
                           style={'text-decoration': 'none', 'color': 'inherit'},
                           href='https://github.com/plotly/dash-svm')), className="banner",),
        # Demo Description
        html.Div(
            className="row background",
            id="demo-explanation",
            style={"padding": "50px 45px"},
            children=[
                html.Div(
                    id="description-text", children=html.H5("This is an active learning tool"
                                                            " which can be used for labelling training "
                                                            "examples, visualizing datasets"
                                                            " and experimenting with Active learning"""
                                                            " and experimenting with Active learning"""
                                                            " visualizing AL algorithms")
                ),
            ]),
        # Body
        html.Div(className="row background", style={"padding" : "10px"}, children=[
                html.Div(
                    className='three columns',
                    style={
                        'display': 'inline-block',
                        'width': '25%',
                        'overflow-y': 'hidden',
                        'overflow-x': 'hidden',
                    },
                    children=[
                        html.P('Select a dataset', style={'text-align': 'left', 'color': 'light-grey'}),
                        dcc.Dropdown(id='select-dataset',
                                     options=[{'label': 'mnist', 'value': 'mnist'},
                                              {'label': 'iris', 'value': 'iris'},
                                              {'label': 'breast-cancer', 'value': 'bc'},
                                              {'label': 'wine', 'value': 'wine'}],
                                     clearable=False,
                                     searchable=False,
                                     value='bc'
                                     ),
                        html.P(' '),
                        html.P('Choose batch size',
                               style={'text-align': 'left', 'color': 'light-grey'}),
                        html.Div(dcc.Slider(id='query-batch-size', min=5, max=100, step=None,
                                   marks={
                                            i: str(i) for i in [5, 10, 50, 100]
                                        }, value=5 ), style={"margin-bottom": "20px"}),



                    ]),
                html.Div(className="six columns", children=[
                    dcc.Loading(dcc.Graph(id='scatter'), fullscreen=True),
                    dcc.Loading(dcc.Graph(id='decision'), fullscreen=True)]),
                html.Div(className="three columns", children=[
                    html.Button('Next round', id='button', autoFocus=True,
                                style={'color': 'white', 'background-color': 'green'}),
                    html.Div([
                        html.H4('Labels'),
                        html.Div(id='label'),
                        dcc.Input(id='query',
                                  placeholder='enter the label',
                                  type='text',
                                  debounce=True,
                                  disabled=True,
                                  value=''),
                        html.Button('Submit', id='submit'),
                    ], ),
                    html.H4('Score'),
                    html.Div(id='score')]),
        ]),
        html.Div(id='selected-data'),
        dcc.Loading(html.Div(id='hidden-div', style={'display': 'none'}), fullscreen=True),

    ])

# Running the server
if __name__ == '__main__':
    app.run_server(debug=True)