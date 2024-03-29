import dash
import dash_html_components as html
import dash_core_components as dcc
from app.callbacks import register_callbacks
import os
import shutil

app = dash.Dash(__name__, url_base_pathname='/dashboard/')
app.config.suppress_callback_exceptions = True
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True


# 1. Layout of the app
app.layout = html.Div(
    children=[
        # TOP BAR AND BANNER
        html.Div(
            id='top-bar',
            className='row',
            style={'backgroundColor': '#fa4f56',
                   'height': '5px',
                   }
        ),
        html.Div(
            html.H4(html.A('Active Learning Explorer',
                           style={'text-decoration': 'none', 'color': 'inherit'},
                           href='https://github.com/plotly/dash-svm'))),
        # CONTROL SECTION FOR USER

        html.Div(
            className='control-section',
            children=[
                html.Div(className='control-element',
                         children=[html.Div(children=["Select Dataset:"],
                                            style={'width': '40%'}),
                                   html.Div(dcc.Dropdown(id='select-dataset',
                                                         options=[{'label': 'davidson', 'value': 'davidson_dataset'},
                                                                  {'label': 'founta', 'value': 'founta_dataset'},
                                                                  {'label': 'gao', 'value': 'gao_dataset'},
                                                                  {'label': 'golbeck', 'value': 'golbeck_dataset'},
                                                                  {'label': 'waseem', 'value': 'waseem_dataset'},
                                                                  {'label': 'mnist-mini', 'value': 'mnist'},
                                                                  # support only text and image for now
                                                                  # {'label': 'iris', 'value': 'iris'},
                                                                  # {'label': 'breast-cancer', 'value': 'bc'},
                                                                  # {'label': 'wine', 'value': 'wine'}
                                                                  ],
                                                         clearable=False,
                                                         searchable=False,
                                                         value='davidson_dataset'
                                                         ), style={'width': '60%'})
                                   ]),
                html.Div(className='control-element',
                         children=[
                             html.Div(children=["Select batch size:"],
                                      style={'width': '40%'}),
                             html.Div(dcc.Slider(id='query-batch-size', min=2, max=100, step=None,
                                                 marks={
                                                     i: str(i) for i in [2, 3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                                                 }, value=2), style={'width': '60%'})]),
                html.Div(className='control-element',
                         children=[
                             html.Div(children=["Active learning method:"],
                                      style={'width': '40%'}),
                             html.Div(dcc.Dropdown(id='al',
                                                   options=[{'label': 'k-means-closest', 'value': 'k-means-closest'},
                                                            {'label': 'k-means-uncertain', 'value': 'k-means-uncertain'},
                                                            {'label': 'ranked-batch-mode', 'value': 'ranked'},
                                                            ],
                                                   value='k-means-closest',
                                                   # style={"display": "none"}
                                                   ), style={'width': '60%'})]),

                html.Div(className='control-element',
                         children=[
                             html.Div(children=["Dimensionality reduction (viz):"],
                                      style={'width': '40%', "display": "none"}),
                             html.Div(dcc.RadioItems(id='dim',
                                                     options=[{'label': 'PCA', 'value': 'pca'},
                                                              {'label': 'T-SNE', 'value': 'tsne'},
                                                              {'label': 'UMAP', 'value': 'umap'}],
                                                     value='pca',
                                                     style={"display": "none"}
                                                     ), style={'width': '60%'})]),
            ]),
        html.Button('Start', id='start', style={'width': '40%'}),

        html.Div(
            id='mid-bar',
            className='row',
            style={'backgroundColor': '#fa4f56',
                   'height': '5px',
                   }
        ),

        # Body



        dcc.Tabs(id='tab', children=[
            dcc.Tab( label="Train batch", children=[
                html.Div([
                html.Div(id="done"),
                html.Div(children=[
                    html.Button('Fetch next batch', id='next_round', autoFocus=True,
                                style={'color': 'white', 'background-color': 'green'}),
                    html.H1(' '),
                    html.Div(id="query_data"),
                    html.H1(' '),
                    html.Div([
                        html.Div(id='label', style={'display': 'none'}),
                        dcc.RadioItems(id='radio_label'),
                        html.Button('Next', id='submit'),
                    ], ),
                    html.Div(id='n_times', style={'display': 'none'}),

                ]),

                    dcc.Loading(html.Div([
                    dcc.Graph(id='scatter', style={'width': '49%', 'display': 'inline-block'}),
                    dcc.Graph(id='ground', style={'width': '45%', 'display': 'inline-block'})])),
                dcc.Graph(id='scatter-hidden', style={'display': 'none'})])


            ]),

            dcc.Tab(label="Batch output",  children=[
                html.Div([
                html.Div([dcc.Graph(id='decision', style={'width': '49%', 'display': 'inline-block'}),
                          dcc.Graph(id='decision1', style={'width': '49%', 'display': 'inline-block'})]),
                html.Div(id='score')])]),
            dcc.Tab(label="View clusters", children=[
                html.Div([dcc.Graph(id='cluster_plot', style={'height': '100%'})]),

            ]),
        ]),




        html.Div(id='selected-data'),
        dcc.Store(id='store'),
        dcc.Store(id='querystore'),
        dcc.Store(id='store_dataset'),
        html.Div(id='start_timer'),

        dcc.Loading(html.Div(id='hidden-div', style={'display': 'none'}), fullscreen=True),

    ])

# 2. Callbacks
register_callbacks(app)


# Running the server
if __name__ == '__main__':
    shutil.rmtree('.cache', ignore_errors=True)
    os.mkdir('.cache')
    app.run_server(debug=True)