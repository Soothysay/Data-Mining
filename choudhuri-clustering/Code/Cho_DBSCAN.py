#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 11:10:08 2022

@author: akashchoudhuri
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import silhouette_score
df=pd.read_csv('cho.txt',sep='\t',header=None)
ground_truth=df[1].tolist()
df1=df[[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]
dims=df1.to_numpy()
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=.5, wspace=.2)
i = 1
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(dims)
# Defining the list of hyperparameters to try
eps_list=np.arange(start=0.1, stop=0.9, step=0.01)
min_sample_list=np.arange(start=2, stop=5, step=1)
 
# Creating empty data frame to store the silhouette scores for each trials
silhouette_scores_data=pd.DataFrame()

for eps_trial in eps_list:
    for min_sample_trial in min_sample_list:
        
        # Generating DBSAN clusters
        db = DBSCAN(eps=eps_trial, min_samples=min_sample_trial)
        
        if(len(np.unique(db.fit_predict(dims)))>1):
            sil_score=silhouette_score(dims, db.fit_predict(dims))
        else:
            continue
        trial_parameters="eps:" + str(eps_trial.round(1)) +" min_sample :" + str(min_sample_trial)
        
        silhouette_scores_data=silhouette_scores_data.append(pd.DataFrame(data=[[sil_score,trial_parameters]], columns=["score", "parameters"]))
 
# Finding out the best hyperparameters with highest Score
silhouette_scores_data.sort_values(by='score', ascending=False).head(1)
silhouette_scores_data.to_csv('silscore.csv')

db = DBSCAN(eps=0.7, min_samples=4)
pred_labs=db.fit_predict(dims)

sns.scatterplot(x=principalComponents[:,0], 
                y=principalComponents[:,1],
                hue=pred_labs,
                palette="rainbow").set_title('Labeled Gene Data Reduced with DBSCAN')

min_samples=dims.shape[1]*2
k=min_samples if min_samples>2 else 2
from sklearn.neighbors import NearestNeighbors
nbrs=NearestNeighbors(n_neighbors=k).fit(dims)
distances,indices=nbrs.kneighbors(dims)
for enum,row in enumerate(distances):
    print([round(x,2) for x in row])
farthest=distances[:,-1]
farthest=np.sort(farthest)[::-1]

plt.plot(farthest)
plt.xlabel('index')
plt.ylabel('Distance')
plt.grid(True)
plt.show()

db = DBSCAN(eps=2.5)
pred_labs=db.fit_predict(dims)

sns.scatterplot(x=principalComponents[:,0], 
                y=principalComponents[:,1],
                hue=pred_labs,
                palette="rainbow").set_title('Labeled Gene Data Reduced with DBSCAN')
from sklearn import metrics
Sil1 = metrics.silhouette_score(dims, pred_labs)
CH1 = metrics.calinski_harabasz_score(dims,pred_labs)
DB1 = metrics.davies_bouldin_score(dims,pred_labs)

from sklearn.metrics.cluster import rand_score
eval1=rand_score(ground_truth,pred_labs)
from sklearn.metrics import jaccard_score
eval2=jaccard_score(ground_truth,pred_labs,average='macro')