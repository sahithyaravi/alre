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

