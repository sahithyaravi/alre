
from .all_imports import *


def register_callbacks(app):
    @app.callback([Output('scatter', 'figure'),
                   Output('label', 'children'),
                   Output('store', 'data'),
                   Output('radio_label', 'options'),
                   Output('n_times', 'value'),],
                  [Input('next_round', 'n_clicks'),
                   Input('start', 'n_clicks')],
                  [State('select-dataset', 'value'),
                  State('query-batch-size', 'value'),
                   State('dim', 'value'),
                   State('store', 'data')])
    def update_scatter_plot(n_clicks, start, dataset, batch_size, dim, storedata):
        df, x, y = get_dataset(dataset)
        if storedata is None:
            storedata = 0
        if start is None:
            start = 0
        if n_clicks is None or n_clicks == 0 or start > storedata:
            n_clicks = 0
            query_indices, x_pool, y_pool = init_active_learner(x, y, batch_size)

        else:
            x_pool = np.load('.cache/x_pool.npy')
            df = pd.read_pickle('.cache/df.pkl')
            learner = pickle.load(open(filename, 'rb'))
            query_indices, query_instance, uncertainity = learner.query(x_pool)

        # Plot the query instances
        principals = visualize(x_pool, dim)
        df_pca = pd.DataFrame(principals, columns=['1', '2'])
        selected = principals[query_indices]
        if n_clicks > 0:
            name = 'query'+str(n_clicks)
        else:
            name = 'init random train set'
        data = [
            go.Scatter(x=df_pca['1'],
                       y=df_pca['2'],
                       mode='markers',
                       marker=dict(color='lightblue'),
                       name='unlabeled data'),
            go.Scatter(x=selected[:, 0],
                       y=selected[:, 1],
                       mode='markers',
                       marker=dict(color='navy', size=15,
                                   line=dict(color='navy', width=12)),
                       name=name)]
        layout = go.Layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            clickmode='event+select')
        fig = go.Figure(data, layout)

        # Labels
        values = np.unique(y)
        labels = df['target'].unique()
        tuple_list = zip(values, labels)
        options = []
        for l, value in tuple_list:
            options.append({'label': l, 'value': value})
        # Save files
        np.save('.cache/selected.npy', x_pool[query_indices])
        df.ix[query_indices].to_pickle('.cache/selected.pkl')
        df.drop(query_indices, inplace=True)
        df = df.reset_index(drop=True)
        df.to_pickle('.cache/df.pkl')
        df_pca.to_pickle('.cache/df_pca.pkl')

        return fig, dataset, start, options, n_clicks

    @app.callback(
        Output('dummy', 'children'),
        [Input('n_times', 'value'),
         Input('submit', 'n_clicks'),],
        [State('select-dataset', 'value')]
        )
    def enable_query(next_round, submit, dataset):
      #  print(next_round, dataset)
        image = " "
        if next_round is None or next_round == 0:
            return image
        if dataset == "mnist":
            try:
                index = 0
                image_vector = (np.load('.cache/selected.npy')[index])
                image_np = image_vector.reshape(8, 8).astype(np.float64)
                image_b64 = numpy_to_b64(image_np)
                image = html.Img(
                    src="data:image/png;base64, " + image_b64,
                    style={"display": "block", "height": "10vh", "margin": "auto"},
                )
            except ValueError:
                pass
        elif dataset == "davidson":
            selected = pd.read_pickle('.cache/selected.pkl')
            try:
                index = 0
                selected = selected.reset_index(drop=True)
            except ValueError:
                pass
            if selected.empty:
                image = ""
            else:
                image = html.Div(html.H6(selected.ix[index]['text']))
                selected.drop(0, inplace=True)
            selected.to_pickle('.cache/selected.pkl')

        return image

    @app.callback(
        [
         Output('hidden-div', 'children'),
         Output('store_dataset', 'data')],
        [Input('scatter', 'selectedData'),
         Input('submit', 'n_clicks'),
         Input('start', 'n_clicks'),
         ],
        [State('store_dataset', 'data'),
         State('hidden-div', 'children'),
         State('radio_label', 'value')])
    def get_selected_data(clickData, submit,  start, store, previous, radio_label):
        if store is None:
            store = 0
        if start is None:
            start = 0

        if previous is None or start > store:
            result_dict = dict()
            result_dict['clicks'] = 0
            #result_dict['points'] = []
            result_dict['queries'] = []
        else:
            if radio_label is not None and previous is not None:
                if submit > literal_eval(previous)["clicks"]:

                    previous_list = json.loads(previous)
                    result_dict = previous_list
                    #points = previous_list['points'] + clickData['points']
                    queries = previous_list['queries']+[int(radio_label)]
                    #result_dict['points'] = points
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
         Input('label', 'children'),
         Input('select-dataset', 'value'),

         ], [State('querystore', 'data')])
    def perform_active_learning(previous, n_clicks, batch_size, labels, dataset, query_round):
        decision = go.Figure()
        score = ''
        colorscale = 'Rainbow'
        if n_clicks is None and labels == dataset:
            x = np.load('.cache/x_pool.npy')
            y = np.load('.cache/y_pool.npy')
            learner = pickle.load(open(filename, 'rb'))
            predictions = learner.predict(x)
            is_correct = (predictions == y)
            score = str(round(learner.score(x, y), 3))
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
                    x_pool = np.load('.cache/x_pool.npy')
                    y_pool = np.load('.cache/y_pool.npy')

                    learner = pickle.load(open(filename, 'rb'))
                    query_results = literal_eval(previous)['queries'][0:batch_size]
                    query_indices = list(range(0, batch_size))
                    learner.teach(x_pool[query_indices], query_results)
                    # Active learner supports numpy matrices, hence use .values
                    df_pca = pd.read_pickle('.cache/df_pca.pkl')
                    predictions = learner.predict(x_pool)
                    is_correct = (predictions == y_pool)
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
                    # Remove query indices from unlabelled pool
                    x_pool = np.delete(x_pool, query_indices, axis=0)
                    y_pool = np.delete(y_pool, query_indices)
                    np.save('.cache/x_pool.npy', x_pool)
                    np.save('.cache/y_pool.npy', y_pool)
                    score = learner.score(x_pool, y_pool)
                    score = ('Batch #' + str(n_clicks)+' Score: ' + str(round(score, 3)))
                    decision = go.Figure(data_dec, layout=layout)
        return decision, score


def get_dataset(dataset):
    if dataset == "mnist":
        raw_data = load_digits()  # fetch_openml('mnist_784', version=1)
        df = pd.DataFrame(data=raw_data['data'])
        df['target'] = raw_data['target']
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
        df['target'] = df['label']
        df.drop('text', axis=1)

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
    #print(df.head())
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


def visualize(x_pool, dim):
    if dim == "pca":
        pca = PCA(n_components=2, random_state=100)
        principals = pca.fit_transform(x_pool)

    # tsne = TSNE(n_components=2, random_state=100, n_iter=300)
    # principals_sne = tsne.fit_transform(x)
    # np.save('.cache/sne.npy',principals_sne)
    # umaps = umap.UMAP(n_components=2, random_state=100)
    # principals_umap= umaps.fit_transform(x)
    # np.save('.cache/umap.npy', principals_umap)
    return principals


def init_active_learner(x, y, batch_size):
    query_indices = np.random.randint(low=0, high=x.shape[0] + 1, size=batch_size)
    np.save('.cache/x.npy', x)
    np.save('.cache/y.npy', y)
    x_train = x[query_indices]
    y_train = y[query_indices]

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
    x_pool = np.delete(x, query_indices, axis=0)
    y_pool = np.delete(y, query_indices, axis=0)

    np.save('.cache/x_pool.npy', x_pool)
    np.save('.cache/y_pool.npy', y_pool)
    return query_indices, x_pool, y_pool