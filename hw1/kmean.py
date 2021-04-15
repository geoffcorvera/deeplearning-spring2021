# %%
import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt('data/510_cluster_dataset.txt')

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

    def updateCentroids(self):
        for s, i in zip(self.clusters, range(self.k)):
            s = np.array(s)
            if s.size > 0:
                newCentroid = np.mean(s, axis=0)
                self.centroids[i] = newCentroid

    def showClusters(self, X):
        colors = ['#f4d35e', '#f95738', '#0d3b66']
        for s, color in zip(self.clusters, colors):
            s = np.array(s)
            plt.scatter(s[:, :1], s[:, 1:], c=color)
        plt.show()

    def train(self, X, niter=10):
        for _ in range(niter):
            model.assignClusters(X)
            model.updateCentroids()
            print(f'iteration {_}')
            model.showClusters(X)


# %%
model = KMeans(data, 3)
model.train(data)
