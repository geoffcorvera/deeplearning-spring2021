# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
class KMeans(object):

    def __init__(self, X, k):
        randomIndices = np.random.choice(X.shape[0], size=k, replace=False)

        self.k = k
        self.centroids = X[randomIndices, :]
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

    def showClusters(self, X, caption):
        colors = ['#f4d35e', '#f95738', '#0d3b66',
                  '#faf0ca', '#ee964b', '#d11149']
        for s, color in zip(self.clusters, colors):
            s = np.array(s)
            plt.scatter(s[:, :1], s[:, 1:], c=color)

        plt.title(caption)
        plt.show()

    def train(self, X, niter=10):
        for _ in range(niter):
            self.assignClusters(X)
            self.updateCentroids()

    def sumOfSquaresErr(self, X):
        results = self.classify(X)
        err = 0
        for k in range(self.k):
            cluster = X[np.where(results == k)]
            err += np.sum(np.linalg.norm(cluster-self.centroids[k], axis=1))
        return err

    def classify(self, X):
        k_errs = np.linalg.norm(X - self.centroids[0], axis=1)
        for k in range(1, self.k):
            k_dist = np.linalg.norm(X-self.centroids[k], axis=1)
            k_errs = np.c_[k_errs, k_dist]

        # select cluster with closest centroid
        return np.apply_along_axis(np.argmin, 1, k_errs)


# %%
from kmeans3d import KM_3D as km3d
img1 = image.imread('data/Kmean_img1.jpg')
model = km3d(img1, k=5)
# model.train(img1)

# %%
data = np.genfromtxt('data/510_cluster_dataset.txt')

def r_trials(k_clusters, r_trials):
    models = [KMeans(data, k_clusters) for _ in range(r_trials)]
    for m in models:
        m.train(data)

    wcss = np.array([m.sumOfSquaresErr(data) for m in models])
    ibest = np.argmin(wcss)
    bestModel = models[ibest]

    caption = f'Sum of Squares Error: {wcss[ibest]}'
    bestModel.showClusters(data, caption)


r_trials(5, 3)

