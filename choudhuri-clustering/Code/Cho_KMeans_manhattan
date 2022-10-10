#Importing required modules
 
import numpy as np
from scipy.spatial.distance import cdist 
 
#Function to implement steps given in previous section
def kmeans(x,k, no_of_iterations):
    idx = np.random.choice(len(x), k, replace=False)
    #Randomly choosing Centroids 
    centroids = x[idx, :] #Step 1
     
    #finding the distance between centroids and all the data points
    distances = cdist(x, centroids ,'euclidean') #Step 2
     
    #Centroid with the minimum Distance
    points = np.array([np.argmin(i) for i in distances]) #Step 3
     
    #Repeating the above steps for a defined number of iterations
    #Step 4
    for _ in range(no_of_iterations): 
        centroids = []
        for idx in range(k):
            #Updating Centroids by taking mean of Cluster it belongs to
            temp_cent = x[points==idx].mean(axis=0) 
            centroids.append(temp_cent)
 
        centroids = np.vstack(centroids) #Updated Centroids 
         
        distances = cdist(x, centroids ,'euclidean')
        points = np.array([np.argmin(i) for i in distances])
         
    return points 

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('cho.txt',sep='\t',header=None)
ground_truth=df[1].tolist()
df1=df[[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]
dims=df1.to_numpy()

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(dims)

#Applying our function
label = kmeans(dims,4,1000)
 
# distortions = []
# inertias = []
# mapping1 = {}
# mapping2 = {}
# K = range(1, 10)
# from sklearn.cluster import KMeans
# from sklearn import metrics
# from scipy.spatial.distance import cdist
# import numpy as np
# import matplotlib.pyplot as plt
  
# for k in K:
#     # Building and fitting the model
#     kmeanModel = KMeans(n_clusters=k).fit(dims)
#     kmeanModel.fit(dims)
  
#     distortions.append(sum(np.min(cdist(dims, kmeanModel.cluster_centers_,
#                                         'euclidean'), axis=1)) / dims.shape[0])
#     inertias.append(kmeanModel.inertia_)
  
#     mapping1[k] = sum(np.min(cdist(dims, kmeanModel.cluster_centers_,
#                                    'euclidean'), axis=1)) / dims.shape[0]
#     mapping2[k] = kmeanModel.inertia_
    
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('Values of K')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method using Distortion')
# plt.show()

#Visualize the results
 
sns.scatterplot(x=principalComponents[:,0], 
                y=principalComponents[:,1],
                hue=label,
                palette="rainbow").set_title('Labeled Gene Data Reduced with KMeans Eucledian')

from sklearn.metrics.cluster import rand_score
for i in range(len(label)):
    label[i]=label[i]+1
eval1=rand_score(ground_truth,label)
from sklearn.metrics import jaccard_score
eval2=jaccard_score(ground_truth,label,average='macro')