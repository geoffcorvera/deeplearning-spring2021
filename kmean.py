# %%
import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt('hw1/510_cluster_dataset.txt')

plt.scatter(data[:, :1], data[:, 1:])
plt.show()


# %%
class KMeans(object):

    def __init__(self, X, k):
        randomIndices = np.random.choice(X.shape[0], size=k, replace=False)

        self.k = k
        self.centroids = data[randomIndices, :]
        self.assignClusters(X)

    # TODO: generate data cells from current clusters (eliminate unneeded arg in displayClusters)

    def assignClusters(self, X):
        newClusters = [list() for _ in range(self.k)]

        for x in X:
            deltas = [x - m for m in self.centroids]
            norms = [np.linalg.norm(d) for d in deltas]
            nearestCluster = np.argmin(norms)
            newClusters[nearestCluster].append(x)

        self.clusters = newClusters
