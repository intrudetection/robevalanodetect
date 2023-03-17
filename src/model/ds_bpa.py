import numpy as np
from sklearn.cluster import KMeans

class BPA:

    
    def __init__(self, E):
        self.__E = E
        self.__interpretation = {0:'Normal', 1:'Uncertain' , 2:'Abnormal'}
        
    def set_class_centroid_error(self):
        E = self.__E.reshape((-1, 1))
        kmeans = KMeans(n_clusters=3, random_state=0).fit(E)
        centroids = kmeans.cluster_centers_.reshape((1, -1))[0]
        centroids.sort()
        self.__centroids = centroids.copy()
    
    def __softmax(self, z):
        assert len(z.shape) == 2
        s = np.max(z, axis=1)
        s = s[:, np.newaxis]
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]
        return e_x / div 
        
    def predict(self, e_x):
        x = np.array([[abs(e_x-c) for c in self.__centroids]])
        pred = self.__softmax(-x)
        return pred, self.__interpretation[np.argmax(pred)]