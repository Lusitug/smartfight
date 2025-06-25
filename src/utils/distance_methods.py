import numpy as np

class DistancesDTW:
    @staticmethod
    def squared_euclidean_distance(x, y):
        return np.sum((x - y) ** 2)

    @staticmethod
    def chebyshev_distance(x, y):
        return np.max(np.abs(x - y))

    @staticmethod
    def manhattan_distance(x, y):
        return np.sum(np.abs(x - y))

    @staticmethod
    def euclidean_distance(x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    @staticmethod
    def cosine_distance(x, y):
        return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

    @staticmethod
    def canberra_distance(x, y):
        return np.sum(np.abs(x - y) / (np.abs(x) + np.abs(y) + 1e-10))
