import dash
import dash_html_components as html
import dash_core_components as dcc
from app.callbacks import register_callbacks
import os
import shutil
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
                                                            "examples, visualizing"
                                                            " and experimenting with Active learning"""
                                                            )
                ),
            ]),
        # Body
        html.Div(className="row background", style={"padding" : "10px"}, children=[
            dcc.Store(id='store'),
            dcc.Store(id='querystore'),
            dcc.Store(id='store_dataset'),
            html.Div(
                className='two columns',
                style={
                    'display': 'inline-block',
                    'overflow-y': 'hidden',
                    'overflow-x': 'auto',
                },
                children=[
                    html.P('Select a dataset', style={'text-align': 'left', 'color': 'light-grey'}),
                    dcc.Dropdown(id='select-dataset',
                                 options=[{'label': 'davidson', 'value': 'davidson'},
                                          {'label': 'mnist-mini', 'value': 'mnist'},
                                          {'label': 'iris', 'value': 'iris'},
                                          {'label': 'breast-cancer', 'value': 'bc'},
                                          {'label': 'wine', 'value': 'wine'}],
                                 clearable=False,
                                 searchable=False,
                                 value='davidson'
                                 ),
                    html.P(' '),
                    html.P('Choose batch size',
                           style={'text-align': 'left', 'color': 'light-grey'}),
                    html.Div(dcc.Slider(id='query-batch-size', min=5, max=100, step=None,
                                        marks={
                                            i: str(i) for i in [5, 10, 50, 100]
                                        }, value=5 ), style={"margin-bottom": "30px"}),
                    html.P('Choose visualization technique',
                           style={'text-align': 'left', 'color': 'light-grey'}),
                    dcc.RadioItems(id='dim',
                                   options=[{'label': 'PCA', 'value': 'pca'},
                                            {'label': 'T-SNE', 'value': 'tsne'},
                                            {'label': 'UMAP', 'value': 'umap'}],
                                   value='pca'
                                   ),
                    html.Button('Start', id='start'),



                ]),
            html.Div(className="eight columns", children=[
                dcc.Loading(dcc.Graph(id='scatter'), fullscreen=True),
                dcc.Loading(dcc.Graph(id='decision'), fullscreen=True),
            ], style={"height": "100%"}),
            html.Div(className="two columns", children=[
                html.Button('Next round', id='next_round', autoFocus=True,
                            style={'color': 'white', 'background-color': 'green'}),
                html.H1(' '),
                html.Div(id="dummy"),
                html.H1(' '),
                html.Div([
                    html.H5('Labels'),
                    html.Div(id='label', style={'display': 'none'}),
                    # dcc.Input(id='query',
                    #           placeholder='enter the label',
                    #           type='text',
                    #           debounce=True,
                    #           disabled=True,
                    #           value=''),
                    dcc.RadioItems(id='radio_label'),
                    html.Button('Submit query', id='submit'),
                ], ),

                html.H4('Score after model fitting:'),
                html.Div(id='score'),
                ]),
        ]),
        html.Div(id='selected-data'),

        dcc.Loading(html.Div(id='hidden-div', style={'display': 'none'}), fullscreen=True),

    ])

# Running the server
if __name__ == '__main__':
    shutil.rmtree('.cache', ignore_errors=True)
    os.mkdir('.cache')
    app.run_server(debug=True)