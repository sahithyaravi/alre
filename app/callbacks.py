from dash.dependencies import Input, Output, State
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.decomposition import PCA
import numpy as np
import plotly.graph_objs as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from functools import partial
from modAL.batch import uncertainty_batch_sampling
from modAL.models import ActiveLearner
import dash_core_components as dcc
import pandas as pd
import pickle
import json
from ast import literal_eval
import dash_html_components as html
filename = 'finalized_model.sav'
cmap_light = [[1, "rgb(165,0,38)"],
              [0.5, "rgb(165,0,38)"],
              [0.45, "rgb(215,48,39)"],
              [0.4, "rgb(244,109,67)"],
              [0.35, "rgb(253,174,97)"],
              [0.3, "rgb(254,224,144)"],
              [0.25, "rgb(224,243,248)"],
              [0.2, "rgb(171,217,233)"],
              [0.15, "rgb(116,173,209)"],
              [0.1, "rgb(69,117,180)"],
              [0.0, "rgb(49,54,149)"]]
cmap_bold = [[0, '#FF0000'], [0.5, '#00FF00'], [1, '#0000FF']]


def register_callbacks(app):
    @app.callback(Output('scatter', 'figure'),

                  [Input('select-dataset', 'value'),
                  Input('query-batch-size', 'value'),
                  Input('button', 'n_clicks'),

                 ])
    def update_scatter_plot(dataset, batch_size, n_clicks):

        if dataset == 'bc':
            raw_data = load_breast_cancer()
        elif dataset == 'iris':
            raw_data = load_iris()
        else:
            raw_data = load_wine()
        df = pd.DataFrame(data=np.c_[raw_data['data'], raw_data['target']],
                          columns=list(raw_data['feature_names']) + ['target'])
        df.to_pickle('df.pkl')

        # Active learner supports numpy matrices, hence use .values
        x = df[raw_data.feature_names].values
        y = df.drop(raw_data.feature_names, axis=1).values
        np.save('x.npy', x)
        np.save('y.npy', y)
        print(np.unique(y))
        print(raw_data.target_names)

        # Define our PCA transformer and fit it onto our raw dataset.
        pca = PCA(n_components=2, random_state=100)
        principals = pca.fit_transform(x)
        df_pca = pd.DataFrame(data=principals, columns=['1', '2'])
        df_pca.to_pickle('df_pca.pkl')

        # Randomly choose training examples
        df_train = df.sample(n=3)
        x_train = df_train[raw_data.feature_names].values
        y_train = df_train.drop(raw_data.feature_names, axis=1).values
        pca_mat = pca.fit_transform(df_train)
        df_train_pca = pd.DataFrame(pca_mat, columns=['1','2'])
        if n_clicks is None or n_clicks == 0:
            # Unlabeled pool
            data = [go.Scatter(x=df_pca['1'],
                               y=df_pca['2'],
                               mode='markers',
                               name='unlabeled data'),
                    go.Scatter(x=df_train_pca['1'],
                               y=df_train_pca['2'],
                               mode='markers',
                               name='labeled data')
                    ]

            df_pool = df[~df.index.isin(df_train.index)]
            x_pool = df_pool[raw_data['feature_names']].values
            y_pool = df_pool.drop(raw_data['feature_names'], axis=1).values.ravel()
            np.save('x_pool.npy', x_pool)
            np.save('y_pool.npy', y_pool)
            # ML model
            rf = RandomForestClassifier(n_jobs=-1, n_estimators=20, max_features=0.8)
            # batch sampling
            preset_batch = partial(uncertainty_batch_sampling, n_instances=3)
            # AL model
            learner = ActiveLearner(estimator=rf,
                                    X_training=x_train,
                                    y_training=y_train.ravel(),
                                    query_strategy=preset_batch)
            predictions = learner.predict(x)
            print(" unqueried score", learner.score(x, y.ravel()))
            data_dec = []
            layout = go.Layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
        else:
            x_pool = np.load('x_pool.npy')
            learner = pickle.load(open(filename, 'rb'))
            query_indices, query_instance, uncertainity = learner.query(x_pool)
            uncertainity = [1 if value > 0.2 else 0 for value in uncertainity]
            # Plot the query instances
            selected = pca.fit_transform(query_instance)
            data = [
                go.Scatter(x=df_pca['1'],
                           y=df_pca['2'],
                               mode='markers',
                               marker=dict(color='grey',
                                           #line=dict(color='grey', width=12)
                                           ),
                               name='unlabeled data'),
                go.Scatter(x=selected[:, 0],
                               y=selected[:, 1],
                               mode='markers',
                               marker=dict(color='black', size=10,
                                           line=dict(color='black', width=12)),
                               name='query'+str(n_clicks)),

                # go.Contour(x=df_pca['1'], y=df_pca['2'],
                #            z=uncertainity,
                #            colorscale=[[0, 'purple'],
                #                        [1, 'cyan'],
                #                        ],
                #            opacity=0.5,
                #            hoverinfo='none',
                #            contours=dict(coloring='fill',
                #                          showlines=False),
                #            showscale=False
                #            )
            ]
            layout = go.Layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                clickmode='event+select'
            )

        pickle.dump(learner, open(filename, 'wb'))
        fig = go.Figure(data, layout)
        return fig

    @app.callback(
        [Output('query', 'disabled'),
         Output('query', 'placeholder')],
        [Input('scatter', 'selectedData')])
    def enable_query(selectedData):
        if selectedData is not None:
            y = np.load('y.npy')
            return False, 'enter labels'+ str(np.unique(y))
        else:
            return True, 'enter label'

    @app.callback(
        Output('hidden-div', 'children'),
        [Input('scatter', 'selectedData'),
         Input('submit', 'n_clicks')],
        [State('query', 'value'),
         State('hidden-div', 'children')])
    def get_selected_data(clickData, submit, query, previous):
        if previous is None:
            print()
            result_dict = dict()
            result_dict['clicks'] = 0
            result_dict['points'] = []
            result_dict['queries']=[]
        else:
            if clickData is not None and query and previous is not None:
                if submit > literal_eval(previous)["clicks"]:

                    previous_list = json.loads(previous)
                    result_dict = previous_list
                    points = previous_list['points'] + clickData['points']
                    queries = previous_list['queries']+[int(query)]
                    result_dict['points'] = points
                    result_dict['clicks'] = submit
                    result_dict['queries'] = queries
                else:
                    result_dict = json.loads(previous)
            else:
                result_dict = json.loads(previous)

        return json.dumps(result_dict)

    @app.callback(
        [Output('decision', 'figure'),
         Output('score', 'children')],
        [Input('hidden-div', 'children'),
         Input('query-batch-size', 'value'),
         ])
    def perform_active_learning(previous, batch_size):
        decision = go.Figure()
        score = ''
        if previous:
            if(literal_eval(previous)["clicks"]) == batch_size:
                print('batch size met')
                x_pool = np.load('x_pool.npy')
                y_pool = np.load('y_pool.npy')
                x = np.load('x.npy')
                y = np.load('y.npy')
                learner = pickle.load(open(filename, 'rb'))
                query_results = literal_eval(previous)['queries']
                print(query_results)
                query_indices = list(range(0, batch_size))
                learner.teach(x_pool[query_indices], query_results)
                # Remove query indices from unlabelled pool
                x_pool = np.delete(x_pool, query_indices, axis=0)
                y_pool = np.delete(y_pool, query_indices)

                # Active learner supports numpy matrices, hence use .values

                df_pca = pd.read_pickle('df_pca.pkl')
                predictions = learner.predict(x)
                data_dec = [go.Scatter(x=df_pca['1'],
                                       y=df_pca['2'],
                                       mode='markers',
                                       name='unlabeled data',
                                       marker=dict(color=predictions,
                                                   colorscale=cmap_bold,
                                                   showscale=True))]
                np.save('x_pool.npy', x_pool)
                np.save('y_pool.npy', y_pool)
                score = learner.score(x, y)
                print('score after query ' + str(score))
                decision = go.Figure(data_dec)
            return decision, score


