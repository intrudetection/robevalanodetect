import numpy as np
from sklearn.cluster import KMeans
from scipy.special import softmax
from sklearn import preprocessing

class BPA:

    def __init__(self, E):
        self.__E = E
        self.__interpretation = {0:'Normal', 1:'Uncertain' , 2:'Abnormal'}
        self.__feature_range = (60, 65)
        
    def set_class_centroid_error(self):
        E = self.__E.reshape((-1, 1))
        kmeans = KMeans(n_clusters=3, random_state=0).fit(E)
        centroids = kmeans.cluster_centers_.reshape((1, -1))[0]
        centroids.sort()
        self.__centroids = centroids.copy()
    
    def getC(self):
    	return self.__centroids
        
    def predict(self, e_x):
        x = np.array([abs(e_x-c) for c in self.__centroids])
        scaler = preprocessing.MinMaxScaler(feature_range = self.__feature_range)
        z = scaler.fit_transform(x.reshape((-1, 1)))
        pred = softmax(-z)
        return pred, self.__interpretation[np.argmax(pred)]
