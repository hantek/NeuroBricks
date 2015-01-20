from sklearn.decomposition import PCA
from layer import Layer


def pca_whiten(data, residual):
    pca_model = PCA(n_components=data.shape[1])
    pca_model.fit(data)
    energy_dist = pca_model.explained_variance_ratio_
    total_energy = sum(energy_dist)
    target_energy = (1 - residual) * total_energy
    
    i = 0
    sum_energy = 0
    while sum_energy < target_energy:
        sum_energy += energy_dist[i]
        i += 1

    pca_model = PCA(n_components=i, whiten=True)
    pca_model.fit(data)
    return pca_model.transform(data), pca_model


class PCA_whiten(Layer):
    def __init__(self):
        """
        TODO: A theano based PCA capable of using GPU.
        """
        pass


class SubtractMean(Layer):
    def __init__(self, n_in, varin=None):
        super(SubtractMean, self).__init__(n_in, n_in, varin=varin)

    def output(self):
        return (self.varin.T - self.varin.mean(axis=1)).T

    def _print_str(self):
        return "    (" + self.__class__.__name__ + ")"


class SubtractMeanAndNormalize(Layer):
    def __init__(self, n_in, varin=None):
        super(SubtractMeanAndNormalize, self).__init__(n_in, n_in, varin=varin)

    def output(self):
        mean_zero = (self.varin.T - self.varin.mean(axis=1)).T
        return (mean_zero.T / (mean_zero.std(axis=1) + 1e-10)).T
    
    def _print_str(self):
        return "    (" + self.__class__.__name__ + ")"
