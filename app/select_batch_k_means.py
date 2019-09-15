from sklearn.cluster import KMeans
from modAL.uncertainty import classifier_entropy
from scipy.spatial.distance import euclidean
from .all_imports import *


def random_sampling(classfier, x_pool, n_instances):
    selection_strategy = "k-means-closest"
    n_clusters = 10
    # Get entropy using predict_proba from classifier
    entropy = classifier_entropy(classfier, x_pool)
    print("entropy min ", min(entropy))
    print("entropy max ", max(entropy))

    # select top 10% of entropy high instances
    # use it as input to k means
    n_cluster_inputs = round(0.1 * x_pool.shape[0])
    indices = (-entropy).argsort()[:n_cluster_inputs]
    min_indices = entropy.argsort()[:n_cluster_inputs]
    cluster_inputs = x_pool[indices]

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=600, n_init=10)
    distances = []  # distance to the closest centroid
    kmeans.fit(cluster_inputs)
    if selection_strategy == "k-means-closest":
        for i, instance in enumerate(cluster_inputs):
            closest_centroid = kmeans.cluster_centers_[kmeans.labels_[i]]
            distance = euclidean(instance, closest_centroid)
            distances.append(distance)
        selection_base_values = distances
    else:
        selection_base_values = entropy
    batch_indices = [None] * n_instances
    batch_values = [None] * n_instances

    for i, index in enumerate(indices):
        cluster_id = kmeans.labels_[i]
        current_value = selection_base_values[i]
        batch_value = batch_values[cluster_id]
        if batch_value is None or batch_value > current_value:
            batch_indices[cluster_id] = index
            batch_values[cluster_id] = current_value

    #query_indices = filter(lambda x: x is not None, batch_indices)
    # print("query_indices")
    # print(batch_indices)

    ## Plot cluster
    color = ['hsl(' + str(h) + ',80%' + ',50%)' for h in np.linspace(0, 330, n_clusters)]
    cluster_data =[]
    pca = PCA(n_components=2, random_state=100)
    principals_pool = pca.fit_transform(x_pool)
    principals = principals_pool[indices]
    principals_rest = np.delete(principals_pool, indices, axis=0)
    heatmap_indices = np.random.randint(low=0, high=x_pool.shape[0], size=100)

    np.append(heatmap_indices, batch_indices)
    np.append(heatmap_indices,min_indices)


    cluster_data.append (go.Scatter(x= principals_rest[:,0],
                                       y=principals_rest[:,1],
                                       mode='markers',
                                       name='unlabelled pool',
                                       marker=dict(color='grey',
                                                   size=5)


                                       ))
    for cluster_id in np.unique(kmeans.labels_):
        cluster_indices = np.where(kmeans.labels_== cluster_id)
        center_index = batch_indices[cluster_id]
        cluster_principals = principals[cluster_indices]
        #print("cluster" , cluster_id, cluster_principals)
        cluster_data.append(go.Scatter(x= cluster_principals[:,0],
                                       y=cluster_principals[:,1],
                                       mode='markers',
                                       marker=dict(color=color[cluster_id],
                                                   size = 10),
                                       name = 'cluster '+ str(cluster_id),
                                       ))
        cluster_data.append(go.Scatter(x=[principals_pool[center_index, 0]],
                                       y=[principals_pool[center_index, 1]],
                                       mode='markers',
                                       marker=dict(color=color[cluster_id],
                                                   size=10,

                                                   line=dict(color='black',width=5)),

                                       ))
    cluster_data.append(go.Contour(x=principals_pool[heatmap_indices][:, 0],
                                   y=principals_pool[heatmap_indices][:, 1],
                                   z=entropy[heatmap_indices],
                                   name='uncertainity map',
                                   connectgaps=True,
                                   showscale=False,
                                   colorscale='Jet',
                                   contours = dict(coloring="fill",
                                                   showlines=False
                                                   )
                                   ))
    # cluster_data.append(go.Heatmap(x=principals_pool[heatmap_indices][:, 0],
    #                                y=principals_pool[heatmap_indices][:, 1],
    #                                z=entropy[heatmap_indices],
    #                                connectgaps=True,
    #                                showscale=False,
    #                                colorscale='Jet',
    #                                zsmooth='best'
    #
    #                                ))

    fig = go.Figure(data=cluster_data)

    plotly.offline.plot(fig, filename="clustering")
    n_samples = len(x_pool)
    query_idx = []
    for i in range(1, n_instances+1):
        query_idx.append(np.random.choice(range(n_samples)))
    return query_idx, x_pool[query_idx], entropy, fig