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
        html.Div(
            html.H2(html.A('Active Learning Explorer',
                           style={'text-decoration': 'none', 'color': 'inherit'},
                           href='https://github.com/plotly/dash-svm')), className="banner",),
        # Demo Description
        html.Div(
            className="row background",
            id="demo-explanation",
            style={"padding": "10px"},
            children=[
                html.Div(
                    id="description-text", children=html.P("Active learning explorer "
                                                           "for experimenting with "
                                                           "batch mode active learning "
                                                           )
                ),
            ]),
        # Body
        html.Div(className="row background", style={"padding": "10px"}, children=[
            dcc.Store(id='store'),
            dcc.Store(id='querystore'),
            dcc.Store(id='store_dataset'),
            dcc.Store(id='start_timer'),
            html.Div(className="eight columns", children=[
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
                html.Div(id='n_times', style={'display': 'none'})
            ]),
                dcc.Loading(dcc.Graph(id='scatter')),
                dcc.Graph(id='scatter-hidden', style={'display': 'none'}),
                html.Div(id='score'),
                dcc.Loading(dcc.Graph(id='decision')),
                dcc.Graph(id='ground'),
            ], ),
            html.Div(
                className='three columns',
                style={
                    'display': 'inline-block',
                    'overflow-y': 'hidden',
                    'overflow-x': 'hidden',
                },
                children=[
                    html.P('Select a dataset', style={'text-align': 'left', 'color': 'light-grey'}),
                    dcc.Dropdown(id='select-dataset',
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
                                 value='gao_dataset'
                                 ),
                    html.P(' '),
                    html.P('Choose batch size',
                           style={'text-align': 'left', 'color': 'light-grey'}),
                    html.Div(dcc.Slider(id='query-batch-size', min=3, max=100, step=None,
                                        marks={
                                            i: str(i) for i in [3, 5, 10, 50, 100]
                                        }, value=3), style={"margin-bottom": "30px"}),
                    html.P('Choose visualization technique',
                           style={'text-align': 'left', 'color': 'light-grey',
                                  "display":"none"}),
                    dcc.Dropdown(id='al',
                                 options=[{'label': 'k-means-closest', 'value': 'k-means-closest'},
                                          {'label': 'k-means-uncertain', 'value': 'k-means-uncertain'},
                                          {'label': 'ranked-batch-mode', 'value': 'ranked'},
                                          ],
                                 value='k-means-closest',
                                 #style={"display": "none"}
                                 ),
                    dcc.RadioItems(id='dim',
                                   options=[{'label': 'PCA', 'value': 'pca'},
                                            {'label': 'T-SNE', 'value': 'tsne'},
                                            {'label': 'UMAP', 'value': 'umap'}],
                                   value='pca',
                                   style={"display": "none"}
                                   ),
                    html.Button('Start', id='start'),

                ]),
        ]),

        html.Div(id='selected-data'),

        dcc.Loading(html.Div(id='hidden-div', style={'display': 'none'}), fullscreen=True),

    ])

# 2. Callbacks
register_callbacks(app)


# Running the server
if __name__ == '__main__':
    shutil.rmtree('.cache', ignore_errors=True)
    os.mkdir('.cache')
    app.run_server(debug=True)