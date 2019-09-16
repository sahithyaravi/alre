from dash.dependencies import Input, Output, State
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.decomposition import PCA
from PIL import Image
from io import BytesIO
import base64
import PIL.ImageOps
import numpy as np
import plotly.graph_objs as go
import plotly.offline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from functools import partial
from modAL.batch import uncertainty_batch_sampling
from modAL.models import ActiveLearner
import dash_core_components as dcc
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
import pickle
import json
from ast import literal_eval
import dash_html_components as html
from sklearn.datasets import fetch_openml
import umap
#from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import time
import sklearn.metrics
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

def plot_cluster(x_pool, batch_indices, indices, entropy, labels_ ):
    n_clusters = pd.Series(labels_).nunique()
    color = ['hsl(' + str(h) + ',80%' + ',50%)' for h in np.linspace(0, 330, n_clusters)]
    cluster_data = []
    pca = PCA(n_components=2, random_state=100)
    principals_pool = pca.fit_transform(x_pool)
    principals = principals_pool[indices]
    principals_rest = np.delete(principals_pool, indices, axis=0)
    heatmap_indices = np.random.randint(low=0, high=x_pool.shape[0], size=250)

    np.append(heatmap_indices, batch_indices)

    cluster_data.append(go.Scatter(x=principals_rest[:, 0],
                                   y=principals_rest[:, 1],
                                   mode='markers',
                                   name='unlabelled pool',
                                   showlegend=False,
                                   marker=dict(color='grey',
                                               size=5)

                                   ))
    for cluster_id in np.unique(labels_):
        cluster_indices = np.where(labels_ == cluster_id)
        center_index = batch_indices[cluster_id]
        cluster_principals = principals[cluster_indices]
        # print("cluster" , cluster_id, cluster_principals
        # )

        cluster_data.append(go.Scattergl(x=cluster_principals[:, 0],
                                       y=cluster_principals[:, 1],
                                       mode='markers',
                                       marker=dict(color=color[cluster_id],
                                                   size=10),
                                       name='cluster ' + str(cluster_id),
                                       ))
        cluster_data.append(go.Scattergl(x=[principals_pool[center_index, 0]],
                                       y=[principals_pool[center_index, 1]],
                                       mode='markers',
                                       showlegend=False,
                                       marker=dict(color=color[cluster_id],
                                                   size=15,
                                                   line=dict(color='black', width=5)),
                                       name='centroid cluster ' + str(cluster_id)

                                       ))
    # cluster_data.append(go.Contour(x=principals_pool[heatmap_indices][:, 0],
    #                                y=principals_pool[heatmap_indices][:, 1],
    #                                z=entropy[heatmap_indices],
    #                                name='uncertainity map',
    #                                connectgaps=True,
    #                                showscale=False,
    #                                colorscale='Jet',
    #                                contours=dict(coloring="fill",
    #                                              showlines=False
    #                                              )
    #                                ))
    cluster_data.append(go.Heatmap(x=principals_pool[heatmap_indices][:, 0],
                                   y=principals_pool[heatmap_indices][:, 1],
                                   z=entropy[heatmap_indices],
                                   connectgaps=True,
                                   showscale=False,
                                   colorscale='Jet',
                                   zsmooth='best'

                                   ))

    fig = go.Figure(data=cluster_data)

    #plotly.offline.plot(fig, filename="clustering.html")
    return fig
