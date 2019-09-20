import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.offline
from sklearn.cluster import KMeans
from modAL.uncertainty import classifier_entropy
from scipy.spatial.distance import euclidean, cosine
from sklearn.decomposition import PCA


def batch_kmeans(classfier, x_pool, n_instances, selection_strategy):
    n_clusters = n_instances
    # Get entropy using predict_proba from classifier
    entropy = classifier_entropy(classfier, x_pool)
    # select top 10% of entropy high instances
    # use it as input to k means
    n_cluster_inputs = round(0.1 * x_pool.shape[0])
    indices = (-entropy).argsort()[:n_cluster_inputs]
    min_indices = entropy.argsort()[:n_cluster_inputs]
    cluster_inputs = x_pool[indices]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, init='k-means++')
    sims = []  # distance to the closest centroid
    kmeans.fit(cluster_inputs)
    if selection_strategy == "k-means-closest":
        for i, instance in enumerate(cluster_inputs):
            closest_centroid = kmeans.cluster_centers_[kmeans.labels_[i]]
            sim = 1 - cosine(instance, closest_centroid)
            sims.append(sim)
        selection_base_values = sims
    else:
        print(selection_strategy)
        selection_base_values = entropy
    batch_indices = [None] * n_instances
    batch_values = [None] * n_instances

    for i, index in enumerate(indices):
        cluster_id = kmeans.labels_[i]
        current_value = selection_base_values[i]
        batch_value = batch_values[cluster_id]
        if batch_value is None or batch_value < current_value:
            batch_indices[cluster_id] = index
            batch_values[cluster_id] = current_value

    n_samples = len(x_pool)
    query_idx = []
    for i in range(1, n_instances+1):
        query_idx.append(np.random.choice(range(n_samples)))
    cluster_fig = plot_cluster(x_pool, batch_indices, indices, entropy, kmeans.labels_)
    return batch_indices, x_pool[batch_indices], entropy, cluster_fig


def plot_cluster(x_pool, batch_indices, indices, entropy, labels_ ):
    n_clusters = pd.Series(labels_).nunique()
    colorscale = [[0, 'mediumturquoise'], [1, 'salmon']]
    color = ['hsl(' + str(h) + ',80%' + ',50%)' for h in np.linspace(0, 330, n_clusters)]
    cluster_data = []
    pca = PCA(n_components=2, random_state=100)
    df = pd.read_pickle('.cache/df.pkl')
    principals_pool = pca.fit_transform(x_pool)
    principals = principals_pool[indices]
    df_indices = df.ix[indices].reset_index()
    principals_rest = np.delete(principals_pool, indices, axis=0)
    heatmap_indices = np.random.randint(low=0, high=x_pool.shape[0], size=100)


    np.append(heatmap_indices, batch_indices)

    cluster_data.append(go.Scatter(x=principals_rest[:, 0],
                                   y=principals_rest[:, 1],
                                   mode='markers',
                                   name='more certain data',
                                   showlegend=True,
                                   marker=dict(color='grey',
                                               opacity=0.5,
                                               size=5)

                                   ))
    for cluster_id in np.unique(labels_):
        cluster_indices = np.where(labels_ == cluster_id)

        center_index = batch_indices[cluster_id]
        cluster_principals = principals[cluster_indices]
        df_cluster = df_indices.ix[cluster_indices]

        # print("cluster" , cluster_id, cluster_principals
        # )

        cluster_data.append(go.Scatter(x=cluster_principals[:, 0],
                                       y=cluster_principals[:, 1],
                                       mode='markers',
                                       showlegend=True,
                                       hovertext=df_cluster['text'].values,
                                       marker=dict(color=color[cluster_id],
                                                   size=10),
                                       name='cluster ' + str(cluster_id),
                                       ))
        cluster_data.append(go.Scatter(x=[principals_pool[center_index, 0]],
                                       y=[principals_pool[center_index, 1]],
                                       hovertext=[df.ix[center_index]['text']],
                                       mode='markers',
                                       showlegend=True,
                                       marker=dict(color=color[cluster_id],
                                                   size=15,
                                                   line=dict(color='black', width=5)),
                                       name='centroid cluster ' + str(cluster_id)

                                       ))
    cluster_data.append(go.Contour(x=principals_pool[heatmap_indices][:, 0],
                                   y=principals_pool[heatmap_indices][:, 1],
                                   z=entropy[heatmap_indices],
                                   name='uncertainity map',
                                   visible='legendonly',
                                   showlegend=True,
                                   connectgaps=True,
                                   showscale=False,
                                   colorscale=colorscale,
                                   line=dict(width=0),
                                   contours=dict(coloring="heatmap",
                                                 showlines=False
                                                 )
                                   ))
    # cluster_data.append(go.Heatmap(x=principals_pool[heatmap_indices][:, 0],
    #                                y=principals_pool[heatmap_indices][:, 1],
    #                                z=entropy[heatmap_indices],
    #                                connectgaps=True,
    #
    #                                showscale=False,
    #                                name='Heatmap',
    #                                colorscale='Viridis',
    #                                zsmooth='best'
    #
    #                                ))

    fig = go.Figure(data=cluster_data)

    #plotly.offline.plot(fig, filename="clustering.html")
    return fig

