#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 23:09:30 2022

@author: akashchoudhuri
"""

import pandas as pd
df=pd.read_csv('cho.txt',sep='\t',header=None)
ground_truth=df[1].tolist()
df1=df[[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]
dims=df1.to_numpy()
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from collections import Counter
import random
from tqdm import tqdm
import numpy as np
from sklearn.cluster import SpectralClustering, AffinityPropagation
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(dims)
clustering = SpectralClustering(n_clusters=6, assign_labels="discretize", random_state=0).fit(dims)
y_pred = clustering.labels_
plt.figure(figsize=(14,6))
plt.title(f'Spectral clustering results ')
plt.scatter(principalComponents[:, 0], principalComponents[:, 1], s=50, c = y_pred);

from sklearn import metrics
Sil1 = metrics.silhouette_score(dims, y_pred)
CH1 = metrics.calinski_harabasz_score(dims,y_pred)
DB1 = metrics.davies_bouldin_score(dims,y_pred)

from sklearn.metrics.cluster import rand_score
eval1=rand_score(ground_truth,y_pred)
from sklearn.metrics import jaccard_score
eval2=jaccard_score(ground_truth,y_pred,average='macro')

# from scipy.spatial.distance import pdist, squareform
# def getAffinityMatrix(coordinates, k = 5):
#     """
#     Calculate affinity matrix based on input coordinates matrix and the numeber
#     of nearest neighbours.
    
#     Apply local scaling based on the k nearest neighbour
#         References:
#     https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
#     """
#     # calculate euclidian distance matrix
#     dists = squareform(pdist(coordinates)) 
    
#     # for each row, sort the distances ascendingly and take the index of the 
#     #k-th position (nearest neighbour)
#     knn_distances = np.sort(dists, axis=0)[k]
#     knn_distances = knn_distances[np.newaxis].T
    
#     # calculate sigma_i * sigma_j
#     local_scale = knn_distances.dot(knn_distances.T)

#     affinity_matrix = dists * dists
#     affinity_matrix = -affinity_matrix / local_scale
#     # divide square distance matrix by local scale
#     affinity_matrix[np.where(np.isnan(affinity_matrix))] = 0.0
#     # apply exponential
#     affinity_matrix = np.exp(affinity_matrix)
#     np.fill_diagonal(affinity_matrix, 0)
#     return affinity_matrix

# affinity_matrix = getAffinityMatrix(dims, k = 5)

# import scipy
# from scipy.sparse import csgraph
# # from scipy.sparse.linalg import eigsh
# from numpy import linalg as LA
# def eigenDecomposition(A, plot = True, topK = 5):
#     """
#     :param A: Affinity matrix
#     :param plot: plots the sorted eigen values for visual inspection
#     :return A tuple containing:
#     - the optimal number of clusters by eigengap heuristic
#     - all eigen values
#     - all eigen vectors
    
#     This method performs the eigen decomposition on a given affinity matrix,
#     following the steps recommended in the paper:
#     1. Construct the normalized affinity matrix: L = D−1/2ADˆ −1/2.
#     2. Find the eigenvalues and their associated eigen vectors
#     3. Identify the maximum gap which corresponds to the number of clusters
#     by eigengap heuristic
    
#     References:
#     https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
#     http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
#     """
#     L = csgraph.laplacian(A, normed=True)
#     n_components = A.shape[0]
    
#     # LM parameter : Eigenvalues with largest magnitude (eigs, eigsh), that is, largest eigenvalues in 
#     # the euclidean norm of complex numbers.
# #     eigenvalues, eigenvectors = eigsh(L, k=n_components, which="LM", sigma=1.0, maxiter=5000)
#     eigenvalues, eigenvectors = LA.eig(L)
    
#     if plot:
#         plt.title('Largest eigen values of input matrix')
#         plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
#         plt.grid()
        
#     # Identify the optimal number of clusters as the index corresponding
#     # to the larger gap between eigen values
#     index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:topK]
#     nb_clusters = index_largest_gap + 1
        
#     return nb_clusters, eigenvalues, eigenvectors

# affinity_matrix = getAffinityMatrix(dims, k = 4)
# k, _,  _ = eigenDecomposition(affinity_matrix)
# print(f'Optimal number of clusters {k}')