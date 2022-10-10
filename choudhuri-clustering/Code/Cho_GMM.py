#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 22:23:27 2022

@author: akashchoudhuri
"""

import pandas as pd
df=pd.read_csv('cho.txt',sep='\t',header=None)
ground_truth=df[1].tolist()
df1=df[[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]
dims=df1.to_numpy()
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import numpy as np
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(dims)
from sklearn.mixture import GaussianMixture
n_components = np.arange(1, 10)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(dims)
          for n in n_components]

# plt.plot(n_components, [m.bic(dims) for m in models], label='BIC')
# plt.plot(n_components, [m.aic(dims) for m in models], label='AIC')
# plt.legend(loc='best')
# plt.xlabel('n_components')


gmm = GaussianMixture(5, covariance_type='full', random_state=0).fit(dims)
labels = gmm.predict(dims)

sns.scatterplot(x=principalComponents[:,0], 
                y=principalComponents[:,1],
                hue=labels,
                palette="rainbow").set_title('Labeled Gene Data Reduced with GMM')
from sklearn import metrics
Sil_g = metrics.silhouette_score(dims, labels)
CH_g = metrics.calinski_harabasz_score(dims, labels)
DB_g = metrics.davies_bouldin_score(dims, labels)

from sklearn.metrics.cluster import rand_score
for i in range(len(labels)):
    labels[i]=labels[i]+1
eval1=rand_score(ground_truth,labels)
from sklearn.metrics import jaccard_score
eval2=jaccard_score(ground_truth,labels,average='macro')

