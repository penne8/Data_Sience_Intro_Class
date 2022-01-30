import random as rnd
import numpy as np
import matplotlib.pyplot as plt


class Kmeans:
    def __init__(self, k, vectors):
        self.k = k
        self.vectors = np.array(vectors)
        self.vectorSize = self.vectors.shape[1]
        self.centroids = []
        self.clusters = []
        for i in range(self.k):
            self.clusters.append(list())
        self.shouldTerminate = False

    def update_clusters(self):
        self.clusters = []
        for i in range(self.k):
            self.clusters.append(list())
        # for every vector
        for i in range(self.vectors.shape[0]):
            distances_from_centroids = []
            # find the distances from every centroid
            for j in range(self.k):
                distances_from_centroids.append(np.linalg.norm(self.vectors[i] - self.centroids[j]))
            # add the vector to the minimal centroid cluster
            closest_centroid_id = distances_from_centroids.index(min(distances_from_centroids))
            self.clusters[closest_centroid_id].append(self.vectors[i])

    def update_centroids(self):
        should_terminate = True
        for i in range(self.k):
            cluster_size = len(self.clusters[i])
            centroid = []
            for j in range(self.vectorSize):
                centroid.append(0)
            for vector in range(cluster_size):
                for j in range(self.vectorSize):
                    centroid[j] += self.clusters[i][vector][j]
            for j in range(self.vectorSize):
                centroid[j] /= cluster_size
            if self.centroids[i] != centroid:
                should_terminate = False
                self.centroids[i] = centroid
            print("finished with cluster ", i)
        self.shouldTerminate = should_terminate

    def run(self):
        # guess initial values
        for i in range(self.k):
            rnd_centroid = []
            for j in range(self.vectorSize):
                rnd_centroid.append(rnd.random())
            self.centroids.append(rnd_centroid)

        count = 0
        # start iterating
        while not self.shouldTerminate:
            self.update_clusters()
            print("updated cluster ", count, " times")
            self.print_average()
            self.update_centroids()
            print("updated centroids ", count, " times")
            count += 1
        print("finished")

    def print_average(self):
        plt.plot(self.centroids)
        plt.show()
