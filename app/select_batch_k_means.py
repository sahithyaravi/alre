import numpy as np
from sklearn.cluster import KMeans
from modAL.uncertainty import classifier_entropy


def random_sampling(classfier, x_pool, n_instances):
    # Get entropy using predict_proba from classifier
    entropy = classifier_entropy(classfier, x_pool)

    # select top 10% of entropy high instances
    # use it as input to k means
    n_cluster_inputs = round(0.1 * x_pool.shape[0])
    indices = (-entropy).argsort()[:n_cluster_inputs]
    cluster_inputs = x_pool[indices]
    kmeans = KMeans(n_clusters=60, random_state=0, max_iter=600, n_init=10)
    kmeans.fit(cluster_inputs)
    print(kmeans.labels_)

    n_samples = len(x_pool)
    query_idx = []
    for i in range(1, n_instances+1):
        query_idx.append(np.random.choice(range(n_samples)))
    return query_idx, x_pool[query_idx], entropy