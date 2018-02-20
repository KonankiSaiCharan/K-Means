
# coding: utf-8

# In[1]:


#Importing the necessary packages
import pandas as pd
import mnist
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import pairwise_distances_argmin
import scipy.misc
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances_argmin_min
from scipy import sparse
from numpy import linalg as LA
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import jaccard_similarity_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans


# In[2]:


#Getting the required Pixels
PixelValuesTrain = mnist.train_images()
PixelValuesTest = mnist.test_images()
#Combining the test and train as KMeans is an unsupervised learning algo
AllPixelValues = np.concatenate((PixelValuesTrain,PixelValuesTest),axis = 0)

#Getting the required labels
LabelValuesTrain = mnist.train_labels()
LabelValuesTest = mnist.test_labels()
#Combining the test and train as KMeans is an unsupervised learning algo
AllLabelValues = np.concatenate((LabelValuesTrain,LabelValuesTest),axis = 0)

#Using the reshape function to change the dimensions
CompressedAllPixelValues = AllPixelValues.reshape((AllPixelValues.shape[0], AllPixelValues.shape[1] * AllPixelValues.shape[2]))


# In[25]:


#Create Clusters
def createClusters(DataToBeClustered, countOfClusters, randomNess, numIterations, Labels, dataType):
    #Using this variable we decide the number of times the loop should iterate so that it does not go 
    #to infinite loop
    iterationCount = range(numIterations)
    
    #Initial centroid selection is made using random selection
    Range = np.random.RandomState(randomNess)
    Index = Range.permutation(DataToBeClustered.shape[0])[:countOfClusters]
    centroids = DataToBeClustered[Index]
    FunctiontobeReduced = list()
    
    for itcount in iterationCount:
        #In this step we group all the points closest to centroids
        Class = pairwise_distances_argmin_min(DataToBeClustered, centroids)
        FunctiontobeReduced.append(sum(Class[1] ** 2))
        
        #Using the above formed groups we calculate the mean
        if dataType == "MNIST":
            latestCentroids = np.array([DataToBeClustered[Class[0] == Index].mean(0)
                                        for Index in range(countOfClusters)])
            
        if dataType == "newsData":
            latestCentroids = np.array([DataToBeClustered[Class[0] == Index].mean(0)
                                for Index in range(countOfClusters)])
            latestCentroids = latestCentroids.reshape((latestCentroids.shape[0], latestCentroids.shape[1] * latestCentroids.shape[2]))
        
        #If the old and the new centroids remain the same then we say that the algo has converged
        if np.all(centroids == latestCentroids):
            break
        centroids = latestCentroids
    OriginalLabels = Labels
    maxCluster = list()
    clusterDensity = list()
    clusterDen = list()
    GIImp = list()
    OriginalLabelsLength = len(OriginalLabels)
    for IndexOfCluster in range(centroids.shape[0]):
        Cap = (Class[0] == IndexOfCluster)
        clusterDensity.append(sum(np.bincount(OriginalLabels[Cap])))
        GI = 0
        for i in range(len((np.bincount(OriginalLabels[Cap])))):
            GI += (((np.bincount(OriginalLabels[Cap]))[i])/sum(np.bincount(OriginalLabels[Cap]))) ** 2
        GIImp.append(1 - GI)
        maximum = np.argmax(np.bincount(OriginalLabels[Cap]))
        maxCluster.append(np.bincount(OriginalLabels[Cap]).max())
    maxCluster_sum = sum(maxCluster)
    purityValue = maxCluster_sum/OriginalLabelsLength
    GiniUnits = 0
    sumValue = 0
    for i in range(centroids.shape[0]):
        sumValue += GIImp[i]*clusterDensity[i]
    GiniValue = sumValue/sum(clusterDensity)
    return centroids, Class, purityValue, GiniValue, FunctiontobeReduced, clusterDensity, maxCluster


# In[29]:


centroidsMNIST, ClassMNIST, purityMNIST, GiniMNIST, objMNIST,c,f = createClusters(CompressedAllPixelValues, 10, randomNess = 15, numIterations = 10000,Labels = AllLabelValues, dataType = "MNIST")


# In[30]:


GiniMNIST


# In[31]:


purityMNIST


# In[ ]:


#Let's increase the number of centroids and check the purity and gini...
#As we can see from the below we can say that the values got better


# In[35]:


centroidsMNIST1, ClassMNIST1, purityMNIST1, GiniMNIST1, objMNIST1,c3,f3 = createClusters(CompressedAllPixelValues, 20, randomNess = 15, numIterations = 10000,Labels = AllLabelValues, dataType = "MNIST")


# In[36]:


GiniMNIST1


# In[37]:


purityMNIST1


# In[7]:


#Obtaining the data required to do analysis on 20 NG
newsData = fetch_20newsgroups(subset='train')
converter = TfidfVectorizer(stop_words='english')
#newsData vectorized for consumption
newsDataConsumable = converter.fit_transform(newsData.data)
AllLabelsValuesnewData = newsData.target


# In[26]:


centroidsNewsData, ClassNewsData, purityNewsData, giniNewsData, objNewsData,c1,f1 = createClusters(newsDataConsumable, 20,numIterations = 10000,randomNess = 32,Labels = AllLabelsValuesnewData, dataType = "newsData")


# In[27]:


giniNewsData


# In[28]:


purityNewsData


# In[38]:


centroidsNewsData, ClassNewsData, purityNewsData, giniNewsData, objNewsData,c1,f1 = createClusters(newsDataConsumable, 10,numIterations = 10000,randomNess = 32,Labels = AllLabelsValuesnewData, dataType = "newsData")


# In[39]:


giniNewsData


# In[40]:


purityNewsData


# In[11]:


#Obtaining the fashion data to perform analysis
def getFashionData(path, kind='train'):
    import numpy as np
    import gzip
    import os
    
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images, labels


# In[12]:


fashionDataValues, fashionDataLabels = getFashionData("C:\\Users\\saich\\Desktop\\UnsupervisedML\\fashion-mnist-master\\data\\fashion", kind='train')


# In[32]:


centroidsFashion, ClassFashion, purityFashion, giniFashion, objFashion, c2,f2 = createClusters(fashionDataValues, 10,numIterations = 10000,randomNess = 32,Labels = fashionDataLabels, dataType = "MNIST")


# In[34]:


purityFashion


# In[33]:


giniFashion


# In[41]:


centroidsFashion, ClassFashion, purityFashion, giniFashion, objFashion, c2,f2 = createClusters(fashionDataValues, 20,numIterations = 10000,randomNess = 32,Labels = fashionDataLabels, dataType = "MNIST")


# In[42]:


purityFashion


# In[43]:


giniFashion


# 1. Hence, we can see that for MNIST, 20NG and Fashion datasets we have calculated purity, gini and objective functions.
# 2. And we have also experimented by changing the number of clusters and it is evident that the purity and gini become better when the cluster count is high and become worse when the cluster count is low.
