import numpy as np
import sys


class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, images, lables):
        #simply remembers all the trainig data
        self.images = images
        self.lables = lables

    def predict(self, test_images):
        # assume that each image is vectorized to 1D
        min_dist = sys.maxint
        for i in range(self.images.shape[0]):
            dist = np.sum(np.abs(self.images[i, :] - test_images))
            if dist < min_dist:
                min_dist = dist
                min_index = i

        return self.lables[min_index]