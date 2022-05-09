import pandas as pd 
from datetime import datetime
#from sentence_transformers import SentenceTransformer
import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import manifold 
#from transformers import AutoModelForSequenceClassification
#from transformers import TFAutoModelForSequenceClassification
#from transformers import AutoTokenizer
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
#import preprocessor as p
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import itertools
from scipy.stats import entropy
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.neighbors import NearestNeighbors

from sklearn import mixture



def get_means():
  avg_care = np.average(df[df["care"]>=4][["X_0","X_1"]].to_numpy(),axis=0)
  avg_harm = np.average(df[df["harm"]>=4][["X_0","X_1"]].to_numpy(),axis=0)
  avg_fairness = np.average(df[df["fairness"]>=4][["X_0","X_1"]].to_numpy(),axis=0)
  avg_cheating = np.average(df[df["cheating"]>=4][["X_0","X_1"]].to_numpy(),axis=0)
  avg_loyalty = np.average(df[df["loyalty"]>=4][["X_0","X_1"]].to_numpy(),axis=0)
  avg_betrayal = np.average(df[df["betrayal"]>=4][["X_0","X_1"]].to_numpy(),axis=0)
  avg_subversion = np.average(df[df["subversion"]>=4][["X_0","X_1"]].to_numpy(),axis=0)
  avg_authority = np.average(df[df["authority"]>=4][["X_0","X_1"]].to_numpy(),axis=0)
  avg_purity = np.average(df[df["purity"]>=4][["X_0","X_1"]].to_numpy(),axis=0)
  avg_degradation = np.average(df[df["degradation"]>=4][["X_0","X_1"]].to_numpy(),axis=0)
  return [avg_authority,avg_subversion,avg_purity,avg_degradation,avg_fairness,avg_cheating,avg_loyalty,avg_betrayal,avg_care,avg_harm]



def mahalanobis(x_mu=None, data=None,cov=None):

    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    return mahal.diagonal()
  
  
def select_axes(vice_axe,virtue_axe):

  df.loc[(df[vice_axe] >= df.MV_count) & (df[virtue_axe] < df.MV_count), 'temp_label'] = 0  
  df.loc[(df[virtue_axe] >= df.MV_count) & (df[vice_axe] < df.MV_count), 'temp_label'] = 1
  df.loc[(df[virtue_axe] >= df.MV_count) & (df[vice_axe] >= df.MV_count), 'temp_label'] = 1 
  
  return df



def calc_NearestNeighbors_score(selected_df,k):
  y = selected_df[selected_df["temp_label"]!=-1]["temp_label"].to_numpy()
  X = selected_df[selected_df["temp_label"]!=-1][["X_0","X_1"]].to_numpy()
  nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
  distances, indices = nbrs.kneighbors(X)
  score= []
  for item in indices:
    l = y[item[0]]
    A = np.sum([1 if y[i]==l else 0 for i in item[1:]])
    score.append(A/k)
  return score

def calc_trustScore(vice, virtue, selected_df):
  trust_model = TrustScore()
  alpha_set = selected_df[(selected_df[virtue]>=4) | (selected_df[vice]>=4)][["X_0","X_1","temp_label"]]
  trust_model.fit(alpha_set[["X_0","X_1"]].to_numpy(), alpha_set["temp_label"].to_numpy())
  X_y = selected_df[selected_df["temp_label"]!=-1][["X_0","X_1","temp_label"]]
  trust_score = trust_model.get_score(X_y[["X_0","X_1"]].to_numpy(), X_y["temp_label"].to_numpy())
  return trust_score


def plot_score(selected_df,score,title):
  plt.figure()
  prob_mean = []
  confidence_Mah = []
  confidence_Euc = []
  prob_entropy = []
  sort_score_idx = np.argsort(selected_df[score].to_numpy())
  percentile_score = np.array_split(sort_score_idx, 10)
  for perc in percentile_score:
    prob_entropy.append(np.mean(df_temp.iloc[perc]["entropy"]))
    prob_mean.append(np.mean(selected_df.iloc[perc]["prob"]))
    confidence_Mah.append(np.mean(selected_df.iloc[perc]["confidence_Mah"]))
    confidence_Euc.append(np.mean(selected_df.iloc[perc]["confidence_Euc"]))

  plt.plot(np.arange(10)+1, prob_entropy, 'c+',label="entropy")
  plt.plot(np.arange(10)+1, prob_mean, 'ro',label="annotators")
  plt.plot(np.arange(10)+1, confidence_Mah, 'g^',label="Mahalanobis")
  plt.plot(np.arange(10)+1, confidence_Euc, 'b*',label="Euclidean")
  plt.ylabel('Confidence Avg')
  plt.xlabel(score)
  plt.title(title)
  plt.legend()
  plt.savefig(score+'_'+title+'.png')
  
def read_data(data_path):
  df = pd.read_csv("data_path",index_col="Unnamed: 0")
  df_temp = df[df.columns[3:-2]]
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
  df["temp_label"] = [-1] * len(df)
  df["confidence_Mah"] = [0] * len(df)  
  df["confidence_Euc"] = [0] * len(df)
  df["trust_score"] = [0] * len(df)
  df["NearestNeighbors_score"] = [0] * len(df)
  df["prob_vice"] = [-1] * len(df)
  df["prob_virtue"] = [-1] * len(df)
  #df["ratio"] = [-1] * len(df)
  df["index"] = df.index
  
  
  df["sum"]=df[df.columns[3:13]].sum(axis=1)
  df = df.drop(df[df["sum"]==0].index,axis=0)
  df["entropy"] = df.apply(lambda row : entropy([row["care"]/row["sum"],
                       row["harm"]/row["sum"],
                       row["fairness"]/row["sum"],
                       row["cheating"]/row["sum"],
                       row["loyalty"]/row["sum"],
                       row["betrayal"]/row["sum"],
                       row["authority"]/row["sum"],
                       row["subversion"]/row["sum"],
                       row["purity"]/row["sum"],
                       row["degradation"]/row["sum"]],base=2),axis = 1)

  df["harm_prob"] = df.apply(lambda row : row["harm"]/row["sum"],axis=1)
  df["care_prob"] = df.apply(lambda row : row["care"]/row["sum"],axis=1)

  df["fairness_prob"] = df.apply(lambda row : row["fairness"]/row["sum"],axis=1)
  df["cheating_prob"] = df.apply(lambda row : row["cheating"]/row["sum"],axis=1)

  df["loyalty_prob"] = df.apply(lambda row : row["loyalty"]/row["sum"],axis=1)
  df["betrayal_prob"] = df.apply(lambda row : row["betrayal"]/row["sum"],axis=1)

  df["authority_prob"] = df.apply(lambda row : row["authority"]/row["sum"],axis=1)
  df["subversion_prob"] = df.apply(lambda row : row["subversion"]/row["sum"],axis=1)
  
  df["purity_prob"] = df.apply(lambda row : row["purity"]/row["sum"],axis=1)
  df["degradation_prob"] = df.apply(lambda row : row["degradation"]/row["sum"],axis=1)
  return df
  
  
if __name__ == '__main__':
  df = read_data("Data_path")
  avgs = get_means()
  average_0 = avgs[1::2]
  average_1 = avgs[::2]
  vice_axes = ["subversion","degradation","cheating","betrayal","harm"]
  virtue_axes = ["authority","purity","fairness","loyalty","care"]
  for vice,virtue,avg_0,avg_1 in zip(vice_axes,virtue_axes,average_0,average_1):
    df = select_axes(vice,virtue)
    print(df[df.temp_label==0].shape)
    print(df[df.temp_label==1].shape)
    print(df[df.temp_label==2].shape)
    X = df[df["temp_label"]!=-1][["X_0","X_1"]].to_numpy()
    mean_0 = np.repeat(np.atleast_2d(avg_0),X.shape[0],axis=0)
    mean_1 = np.repeat(np.atleast_2d(avg_1),X.shape[0],axis=0)
    df.loc[df["temp_label"]!=-1,"trust_score"] = np.round(calc_trustScore(vice,virtue, df).astype(np.double),2)
    df.loc[df["temp_label"]!=-1,"NearestNeighbors_score"] = calc_NearestNeighbors_score(df,7)
    df.loc[df["temp_label"]!=-1,"prob_vice"] = np.exp(-mahalanobis(X- mean_0,df[df["temp_label"]!=-1][["X_0","X_1"]]))
    df.loc[df["temp_label"]!=-1,"prob_virtue"] = np.exp(-mahalanobis(X- mean_1,df[df["temp_label"]!=-1][["X_0","X_1"]]))
  
    df.loc[df["temp_label"]==0,"confidence_Mah"] = df[df["temp_label"]==0]["prob_vice"].tolist()
    df.loc[df["temp_label"]==1,"confidence_Mah"] = df[df["temp_label"]==1]["prob_virtue"].tolist() 
    dist_comp0 = np.linalg.norm(X- mean_0,axis=1)
    dist_comp1 = np.linalg.norm(X- mean_1,axis=1)
    df.loc[df["temp_label"]!=-1,"prob_vice"] = np.exp(-dist_comp0)
    df.loc[df["temp_label"]!=-1,"prob_virtue"] = np.exp(-dist_comp1)
    df.loc[df["temp_label"]==0,"confidence_Euc"] = df[df["temp_label"]==0]["prob_vice"].tolist()
    df.loc[df["temp_label"]==1,"confidence_Euc"] = df[df["temp_label"]==1]["prob_virtue"].tolist()
    f_temp = df[df["temp_label"]!=-1]
    df_temp = df_temp[df_temp["trust_score"]<=20]
    df_temp.loc[df_temp["temp_label"]==0,"prob"] = df_temp[df_temp["temp_label"]==0][vice+"_prob"].tolist()
    df_temp.loc[df_temp["temp_label"]==1,"prob"] = df_temp[df_temp["temp_label"]==1][virtue+"_prob"].tolist()
    plot_score(df_temp,'NearestNeighbors_score',vice+"_"+virtue)
    df["temp_label"] = [-1] * len(df)
