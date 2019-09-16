from sklearn.cluster import KMeans
from modAL.uncertainty import classifier_entropy
from scipy.spatial.distance import euclidean
from .all_imports import *


def batch_kmeans(classfier, x_pool, n_instances):
    selection_strategy = "k-means-closest"
    n_clusters = 10
    # Get entropy using predict_proba from classifier
    entropy = classifier_entropy(classfier, x_pool)
    # select top 10% of entropy high instances
    # use it as input to k means
    n_cluster_inputs = round(0.1 * x_pool.shape[0])
    indices = (-entropy).argsort()[:n_cluster_inputs]
    min_indices = entropy.argsort()[:n_cluster_inputs]
    cluster_inputs = x_pool[indices]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=600, init='k-means++')
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

    n_samples = len(x_pool)
    query_idx = []
    for i in range(1, n_instances+1):
        query_idx.append(np.random.choice(range(n_samples)))
    return batch_indices, x_pool[batch_indices], entropy, kmeans.labels_, indices

