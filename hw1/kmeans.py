# %% [markup]
# # K-Means

# %%
import matplotlib.pyplot as plt 
import numpy as np 

data = np.loadtxt('510_cluster_dataset.txt')

# %%
class KMeans(object):
    def __init__(self,X):
        nfeatures = X.shape[1]

    def train(self, X, niter=100):
        for i in range(niter):
            print(i)
    
    def classify(self, X):
        return 1

model = KMeans(data)
model.train(data)
model.classify(data)