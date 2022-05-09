import pandas as pd 
from datetime import datetime
#from sentence_transformers import SentenceTransformer
import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import manifold 
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import preprocessor as p
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import itertools
from scipy.stats import entropy
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch.nn as nn
from sklearn import mixture
from sentence_transformers import SentenceTransformer, models
from torch import nn


def read_data(data_path):
  df = pd.read_csv(data_path,index_col="Unnamed: 0")
  df_temp = df[df.columns[3:-2]]
  #df_temp = df[df.columns[2:-4]]
  df_temp["care"] = pd.to_numeric(df_temp["care"])
  df_temp["harm"] = pd.to_numeric(df_temp["harm"])
  df_temp["fairness"] = pd.to_numeric(df_temp["fairness"])
  df_temp["cheating"] = pd.to_numeric(df_temp["cheating"])
  df_temp["loyalty"] = pd.to_numeric(df_temp["loyalty"])
  df_temp["betrayal"] = pd.to_numeric(df_temp["betrayal"])
  df_temp["authority"] = pd.to_numeric(df_temp["authority"])
  df_temp["subversion"] = pd.to_numeric(df_temp["subversion"])
  df_temp["purity"] = pd.to_numeric(df_temp["purity"])
  df_temp["degradation"] = pd.to_numeric(df_temp["degradation"])
  maxValueIndexObj = df_temp.idxmax(axis=1)
  for axe in set(maxValueIndexObj):
    df.loc[maxValueIndexObj[maxValueIndexObj==axe].index,"MV_count"] = df.loc[maxValueIndexObj[maxValueIndexObj==axe].index.to_list()][axe]
    df.loc[maxValueIndexObj[maxValueIndexObj==axe].index,"MV"] = axe
  df["sum"]=df[df.columns[3:13]].sum(axis=1)
  df = df.drop(df[df["sum"]==0].index,axis=0)
  return df

def sentiment_embedding_model():
  
  task='sentiment'
  #task = 'hate'
  #task = 'offensive'
  MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
  word_embedding_model = models.Transformer(MODEL, max_seq_length=120)
  pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
  dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())

  model_transformer = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
  return model_transformer

def mds_embedding(test_embedding):
  dists=cosine_distances(test_embedding)
  mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=60, max_iter=90000)
  test_embedding_transform = mds.fit(dists)
  return test_embedding_transform


df =read_data("/content/drive/MyDrive/phd_docs/baltimore_BLM_ALM.csv")
model_transformer=sentiment_embedding_model()
embeddings_transform = model_transformer.encode(sentences, show_progress_bar=True)
sentiment_embedding=mds_embedding(embeddings_transform)
df["X_0_fine_tune"] = sentiment_embedding.embedding_[:,0]
df["X_1_fine_tune"] = sentiment_embedding.embedding_[:,1]

fig = px.scatter(df, x="X_0_fine_tune", y="X_1_fine_tune", color="MV", opacity=0.8,hover_data=["prep_text","MV_count"])
fig.show()
fig.write_image("MV_coloring_fine_tune.pdf")
