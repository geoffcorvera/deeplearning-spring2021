import numpy as np

class KM_3D(object):

    def __init__(self, X, k):
        # get random 2D indices
        height, width, dim = X.shape
        mask = np.random.choice((height, width), size=k, replace=False)

        self.k = k
        # self.centroids = X[]
        self.mask = mask