#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 00:46:52 2022

@author: akashchoudhuri
"""

import pandas as pd
df=pd.read_csv('iyer.txt',sep='\t',header=None)
ground_truth=df[1].tolist()
df1=df[[2,3,4,5,6,7,8,9,10,11,12,13]]
dims=df1.to_numpy()
for i in range(len(ground_truth)):
    if ground_truth[i]==-1:
        ground_truth[i]=11
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(dims)
sns.scatterplot(x=principalComponents[:,0], 
                y=principalComponents[:,1],
                hue=ground_truth,
                palette="rainbow").set_title('Labeled Gene Data Reduced with PCA Original')
#Metric 1
clusters = shc.linkage(dims, 
            method='ward', 
            metric="euclidean")
shc.dendrogram(Z=clusters)
plt.axhline(y = 19, color = 'black', linestyle = '-')
plt.show()

clustering_model_pca1 = AgglomerativeClustering(n_clusters=11, affinity='euclidean', linkage='ward')
clustering_model_pca1.fit(principalComponents)

data_labels_pca1 = clustering_model_pca1.labels_
sns.scatterplot(x=principalComponents[:,0], 
                y=principalComponents[:,1],
                hue=data_labels_pca1,
                palette="rainbow").set_title('Labeled Gene Data Reduced with PCA Method 1')
from sklearn import metrics
Sil_g = metrics.silhouette_score(dims, ground_truth)
CH_g = metrics.calinski_harabasz_score(dims, ground_truth)
DB_g = metrics.davies_bouldin_score(dims, ground_truth)

from sklearn.metrics.cluster import rand_score
for i in range(len(data_labels_pca1)):
    data_labels_pca1[i]=data_labels_pca1[i]+1
eval1=rand_score(ground_truth,data_labels_pca1)
from sklearn.metrics import jaccard_score
eval2=jaccard_score(ground_truth,data_labels_pca1,average='macro')

Sil1 = metrics.silhouette_score(dims, data_labels_pca1)
CH1 = metrics.calinski_harabasz_score(dims,data_labels_pca1)
DB1 = metrics.davies_bouldin_score(dims,data_labels_pca1)

# # Metric 2
clusters = shc.linkage(dims, 
            method='complete', 
            metric="minkowski")
shc.dendrogram(Z=clusters)
plt.axhline(y = 10, color = 'black', linestyle = '-')
plt.show()

clustering_model_pca2 = AgglomerativeClustering(n_clusters=11, affinity='minkowski', linkage='complete')
clustering_model_pca2.fit(principalComponents)

data_labels_pca2 = clustering_model_pca2.labels_
sns.scatterplot(x=principalComponents[:,0], 
                y=principalComponents[:,1],
                hue=data_labels_pca2,
                palette="rainbow").set_title('Labeled Gene Data Reduced with PCA Method 2')

for i in range(len(data_labels_pca2)):
    data_labels_pca2[i]=data_labels_pca2[i]+1
    
eval3=rand_score(ground_truth,data_labels_pca2)
eval4=jaccard_score(ground_truth,data_labels_pca2,average='macro')

Sil2 = metrics.silhouette_score(dims, data_labels_pca2)
CH2 = metrics.calinski_harabasz_score(dims,data_labels_pca2)
DB2 = metrics.davies_bouldin_score(dims,data_labels_pca2)