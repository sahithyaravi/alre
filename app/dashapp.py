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
        html.Div(children=[
            html.Div(

                children=dcc.Loading(dcc.Graph(id='scatter'), fullscreen=True),

                style={'width': '60%', 'height': '100%', 'display': 'inline-block', 'position': 'relative'}
            ),



            html.Div(

                style={'width': '20%', 'display': 'inline-block',
                       'position': 'absolute'},
                children=[
                    html.P('Select a dataset', style={'text-align': 'left', 'color': 'light-grey'}),
                    dcc.Dropdown(id='select-dataset',
                                 options=[
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
                    html.Div(dcc.Slider(

                        id='query-batch-size',
                        min=1,
                        max=5,
                        step=1,
                        marks={
                            1: '1',
                            2: '2',
                            3: '3',
                            4: '4',
                            5: '5'},
                        value=3
                    )),


                ]),
            html.Div([dcc.Input(id='query',
                                placeholder='enter the label',
                                type='text',
                                debounce=True,
                                disabled=True,
                                value=''),
                      html.Button('Submit', id='submit'),
                      ],
                     style={'width': '60%', 'height': '100%', 'display': 'inline-block', 'position': 'relative'}),

            html.Div(
                id='decision-graph',
                children=dcc.Loading(dcc.Graph(id='decision'), fullscreen=True),
                style={'width': '60%', 'height': '100%', 'display': 'inline-block', 'position': 'relative'}
            ),
            html.Button('Next round', id='button'),
            html.H4('Score'),
            html.Div(id='score'),
            html.Div(id='selected-data'),
            dcc.Loading(html.Div(id='hidden-div', style={'display': 'none'}), fullscreen=True),
        ])
    ])

# Running the server
if __name__ == '__main__':
    app.run_server(debug=True)