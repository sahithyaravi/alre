from dash.dependencies import Input, Output, State
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.decomposition import PCA
from PIL import Image
from io import BytesIO
import base64
import PIL.ImageOps
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
from sklearn.datasets import fetch_openml
import umap
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import nltk
from nltk.corpus import stopwords

filename = '.cache/finalized_model.sav'
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
    @app.callback([Output('scatter', 'figure'),
                   Output('label', 'children'),
                   Output('store', 'data'),],
                  [Input('next_round', 'n_clicks'),
                   Input('start', 'n_clicks')],
                  [State('select-dataset', 'value'),
                  State('query-batch-size', 'value'),
                   State('dim', 'value'),
                   State('store','data')])
    def update_scatter_plot(n_clicks, start, dataset, batch_size, dim, storedata):
        if storedata is None:
            storedata = 0
        if start is None:
            start = 0
        uncertainity = []
        df, x, y = get_dataset(dataset)
        # Active learner supports numpy matrices, hence use .values

        if n_clicks is None or n_clicks == 0 or start > storedata:
            # Define our PCA transformer and fit it onto our raw dataset.
            # Randomly choose initial training examples
            query_indices = np.random.randint(low=0, high=x.shape[0] + 1, size=batch_size)
            x_pool = np.delete(x, query_indices, axis=0)
            y_pool = np.delete(y, query_indices, axis=0)

            np.save('.cache/x.npy', x)
            np.save('.cache/y.npy', y)
            x_train = x[query_indices]
            y_train = y[query_indices]
            n_clicks = 0
            np.save('.cache/x_pool.npy', x_pool)
            np.save('.cache/y_pool.npy', y_pool)
            x_pool = x
            # ML model
            rf = RandomForestClassifier(n_jobs=-1, n_estimators=20, max_features=0.8)
            # batch sampling
            preset_batch = partial(uncertainty_batch_sampling, n_instances=batch_size)
            # AL model
            learner = ActiveLearner(estimator=rf,
                                    X_training=x_train,
                                    y_training=y_train.ravel(),
                                    query_strategy=preset_batch)
            pickle.dump(learner, open(filename, 'wb'))
            predictions = learner.predict(x)
            print(" unqueried score", learner.score(x, y.ravel()))
            layout = go.Layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                )
            # Viz
            tsne = TSNE(n_components=2, random_state=100, n_iter=300)
            principals_sne = tsne.fit_transform(x)
            np.save('.cache/sne.npy',principals_sne)
            pca = PCA(n_components=2, random_state=100)
            principals_pca = pca.fit_transform(x)
            np.save('.cache/pca.npy', principals_pca)
            umaps = umap.UMAP(n_components=2, random_state=100)
            principals_umap= umaps.fit_transform(x)
            np.save('.cache/umap.npy', principals_umap)
        else:
            x_pool = np.load('.cache/x_pool.npy')
            learner = pickle.load(open(filename, 'rb'))
            query_indices, query_instance, uncertainity = learner.query(x_pool)
            uncertainity = [1 if value > 0.2 else 0 for value in uncertainity]
            layout = go.Layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                clickmode='event+select')
            # Plot the query instances
        np.save('.cache/selected.npy', x_pool[query_indices])
        df.ix[query_indices].to_pickle('.cache/selected.pkl')
        if dim =='pca':
            principals = np.load('.cache/pca.npy')
        elif dim =="tsne":
            principals = np.load('.cache/sne.npy')
        else:
            principals = np.load('.cache/umap.npy')
        df_pca = pd.DataFrame(principals, columns =['1','2'])
        selected = principals[query_indices]
        if n_clicks > 0:
            name = 'query'+str(n_clicks)
        else:
            name = 'init random train set'
        data = [
                go.Scatter(x=df_pca['1'],
                           y=df_pca['2'],
                               mode='markers',
                               marker=dict(color='lightblue',
                                           #line=dict(color='grey', width=12)
                                           ),
                               name='unlabeled data'),
                go.Scatter(x=selected[:, 0],
                               y=selected[:, 1],
                               mode='markers',
                               marker=dict(color='navy', size=15,
                                           line=dict(color='navy', width=12)),
                               name=name),

                # go.Heatmap(x=df_pca['1'],
                #            y=df_pca['2'],
                #            z=uncertainity,
                #            colorscale='RdBu',
                #           opacity=0.5,
                #            hoverinfo='none',
                #            #zsmooth='fast',
                #           showscale=False
                #            ),

            ]
        df_pca.to_pickle('.cache/df_pca.pkl')
        fig = go.Figure(data, layout)
        # Labels
        values = (np.unique(y))
        # try:
        #     names = raw_data.target_names
        # except AttributeError:
        names = values
        label = ' '
        for value in values:
            label = label + str(str(value)+ ' : ' + str(names[int(value)]))+"\n"
        return fig, label, start

    @app.callback(
        [Output('query', 'disabled'),
         Output('query', 'placeholder'),
         Output('dummy', 'children')],
        [Input('scatter', 'selectedData'),
         Input('select-dataset', 'value')])
    def enable_query(selectedData, dataset):
        image = " "
        if selectedData is not None:
            y = np.load('.cache/y.npy')

            if dataset == "mnist" and selectedData["points"][0]["curveNumber"] == 1:
                try:
                    index = selectedData["points"][0]["pointIndex"]
                    image_vector = (np.load('.cache/selected.npy')[index])
                    image_np = image_vector.reshape(8, 8).astype(np.float64)
                    image_b64 = numpy_to_b64(image_np)
                    image = html.Img(
                        src="data:image/png;base64, " + image_b64,
                        style={"display": "block", "height": "10vh", "margin": "auto"},
                    )
                except ValueError:
                    pass
            elif dataset == "davidson" and selectedData["points"][0]["curveNumber"] == 1:
                try:
                    index = selectedData["points"][0]["pointIndex"]
                    selected = pd.read_pickle('.cache/selected.pkl').reset_index()
                    image = html.Div(html.H6(selected.ix[index]['text']))
                except ValueError:
                    pass

            return False, 'enter labels' + str(np.unique(y)), image
        else:
            return True, 'enter label', image

    @app.callback(
        [
          Output('hidden-div', 'children'),
        Output('store_dataset', 'data')],
        [Input('scatter', 'selectedData'),
         Input('submit', 'n_clicks'),
         Input('start', 'n_clicks'),
         ],
        [State('store_dataset', 'data'),
         State('query', 'value'),
         State('hidden-div', 'children')])
    def get_selected_data(clickData, submit,  start, store, query, previous,):
        if store is None:
            store = 0
        if start is None:
            start = 0

        if previous is None or start > store:
            result_dict = dict()
            result_dict['clicks'] = 0
            result_dict['points'] = []
            result_dict['queries'] = []
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

        return json.dumps(result_dict), start

    @app.callback(
        [Output('decision', 'figure'),
         Output('score', 'children'),
         ],
        [Input('hidden-div', 'children'),
         Input('next_round', 'n_clicks'),
         Input('query-batch-size', 'value'),
         Input('label', 'children')

         ],[State('querystore','data')])
    def perform_active_learning(previous, n_clicks, batch_size, labels, query_round):
        decision = go.Figure()
        score = ''
        colorscale = 'Rainbow'
        print(previous)

        if n_clicks is None and labels is not None:
                x = np.load('.cache/x.npy')
                y = np.load('.cache/y.npy')
                learner = pickle.load(open(filename, 'rb'))
                predictions = learner.predict(x)
                is_correct = (predictions == y)
                score = str(round(learner.score(x, y),3))
                df_pca = pd.read_pickle('.cache/df_pca.pkl')
                data_dec = [go.Scatter(x=df_pca['1'].values[is_correct],
                                       y=df_pca['2'].values[is_correct],
                                       mode='markers',
                                       name='correct predictions',
                                       marker=dict(color=predictions[is_correct],
                                                   colorscale=colorscale,
                                                   opacity=0.7,
                                                   showscale=True)),

                            go.Scatter(x=df_pca['1'].values[~is_correct],
                                       y=df_pca['2'].values[~is_correct],
                                       mode='markers',
                                       name='wrong predictions',
                                       marker=dict(symbol="x",
                                                   opacity=0.7,
                                                   colorscale=colorscale,
                                                   color=predictions[~is_correct]))]
                layout = go.Layout(title='Output of classifier', showlegend=False)
                decision = go.Figure(data_dec, layout=layout)
        else:
            if previous and n_clicks is not None:
                if(literal_eval(previous)["clicks"]) == (batch_size*n_clicks):
                    print('batch size met')
                    x_pool = np.load('.cache/x_pool.npy')
                    y_pool = np.load('.cache/y_pool.npy')
                    x = np.load('.cache/x.npy')
                    y = np.load('.cache/y.npy')
                    learner = pickle.load(open(filename, 'rb'))
                    query_results = literal_eval(previous)['queries'][0:batch_size]
                    query_indices = list(range(0, batch_size))
                    learner.teach(x_pool[query_indices], query_results)
                    # Remove query indices from unlabelled pool
                    x_pool = np.delete(x_pool, query_indices, axis=0)
                    y_pool = np.delete(y_pool, query_indices)

                    # Active learner supports numpy matrices, hence use .values

                    df_pca = pd.read_pickle('.cache/df_pca.pkl')
                    predictions = learner.predict(x)
                    is_correct = (predictions == y)
                    data_dec = [go.Scatter(x=df_pca['1'].values[is_correct],
                                           y=df_pca['2'].values[is_correct],
                                           mode='markers',
                                           name='correct predictions',
                                           marker=dict(color=predictions[is_correct],
                                                       colorscale=colorscale,
                                                       opacity=0.7,
                                                       showscale=True)),

                                go.Scatter(x=df_pca['1'].values[~is_correct],
                                           y=df_pca['2'].values[~is_correct],
                                           mode='markers',
                                           name='wrong predictions',
                                           marker=dict(symbol="x",
                                                       colorscale=colorscale,
                                                       opacity=0.7,
                                                       color=predictions[~is_correct]))]
                    layout = go.Layout(title='Output of classifier', showlegend=False)
                    np.save('.cache/x_pool.npy', x_pool)
                    np.save('.cache/y_pool.npy', y_pool)
                    score = learner.score(x, y)
                    score = ('Query#'+ str(n_clicks)+' ' + str(round(score, 3)))
                    print(score)
                    decision = go.Figure(data_dec, layout=layout)

        return decision, score


def get_dataset(dataset):
    if dataset == "mnist":
        raw_data = load_digits()  # fetch_openml('mnist_784', version=1)
        df = pd.DataFrame(data=np.c_[raw_data['data'], raw_data['target']])
        x = raw_data['data']
        y = raw_data['target'].astype(np.uint8)

    elif dataset == "davidson":
        df = pd.read_csv('datasets/davidson_dataset.csv')
        x = df['text'].values
        y = df['label'].values
        nltk.download('stopwords')
        tfid = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7,
                               stop_words=stopwords.words('english'))
        x = tfid.fit_transform(x).toarray()

        print(type(x), type(y))

    else:
        if dataset == 'bc':
            raw_data = load_breast_cancer()
        elif dataset == 'iris':
            raw_data = load_iris()
        elif dataset == "wine":
            raw_data = load_wine()
        df = pd.DataFrame(data=np.c_[raw_data['data'], raw_data['target']],
                          columns=list(raw_data['feature_names']) + ['target'])
        x = raw_data['data']
        y = raw_data['target'].astype(np.uint8)
    return df, x, y


def numpy_to_b64(array, scalar=True):
    # Convert from 0-1 to 0-255
    if scalar:
        array = np.uint8(255 * array)

    im_pil = Image.fromarray(array)
    buff = BytesIO()
    im_pil.save(buff, format="png")
    im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return im_b64


