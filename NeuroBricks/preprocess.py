import sys
import copy
import numpy
import theano
import theano.tensor as T
from sklearn.decomposition import PCA
from layer import Layer, LinearLayer

import pdb

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


class PCA(object):
    class NeuralizedPCALayer(Layer):
        def __init__(self, n_in, n_out, init_w, init_bvis,
                     varin=None, npy_rng=None):
            """
            The difference between a neuralized PCA Layer and a normal linear
            layer is the biases are on the visible side, and the weights are
            initialized as PCA transforming matrix.
            """
            super(PCA.NeuralizedPCALayer, self).__init__(n_in, n_out, varin=varin)
            if not npy_rng:
                npy_rng = numpy.random.RandomState(123)
            assert isinstance(npy_rng, numpy.random.RandomState)

            if (not init_w) or (not init_bvis):
                raise TypeError("You should specify value for init_w and " + \
                                "init_bvis while instantiating this object.")
            # else: TODO assert that they are of valid type.
            
            self.w = init_w
            self.bvis = init_bvis
            self.params = [self.w, self.bvis]

        def fanin(self):
            return T.dot(self.varin - self.bvis, self.w)

        def output(self):
            return self.fanin()

        def activ_prime(self):
            return 1.


    def __init__(self, retain=None):
        """
        A theano based PCA capable of using GPU.
        """
        self.retain = retain

    def fit(self, data, batch_size=10000, verbose=True, whiten=False):
        """
        Part of the code is adapted from Roland Memisevic's code.
        fit() establishes 2 LinearLayer objects: PCAForwardLayer and
        PCABackwardLayer. They define how the data is mapped after the PCA
        mapping is learned.
        """
        assert isinstance(data, numpy.ndarray), \
               "data has to be a numpy ndarray."
        data = data.copy().astype(theano.config.floatX)
        ncases, self.ndim = data.shape
        nbatches = (ncases + batch_size - 1) / batch_size
        data_batch = T.matrix('data_batch')
        
        # centralizing data
        """
        If you don\'t centralize the dataset, then you are still going to get
        perfect reconstruction from the forward/backward mapping matrices, but
        1. the eigenvalues you get will no longer match the variance of each
           principle components, 
        2. the \'principle component\' you get will no longer match the
           projection of largest variance, and
        3. the output will not be centered at the initial data center, neither
           at the origin too. However, the shape of the data scatter would
           still remain intact.
        It just rotates the data by an unwanted angle and shifts the data by an
        unexpected vector.
        """
        np_ncases = numpy.array([ncases]).astype(theano.config.floatX)
        fun_batchmean = theano.function(
            inputs=[data_batch], outputs=T.sum(data_batch / np_ncases, axis=0)
        )
        if verbose:     print "Centralizing data, %d dots to punch:" % nbatches,
        self.mean = numpy.zeros(self.ndim, dtype=theano.config.floatX)
        for bidx in range(nbatches):
            if verbose:
                print ".",
                sys.stdout.flush()
            start = bidx * batch_size
            end = min((bidx + 1) * batch_size, ncases)
            self.mean += fun_batchmean(data[start:end, :])
        data -= self.mean
        if verbose:     print "Done."
        
        # compute convariance matrix
        covmat = theano.shared(
            value=numpy.zeros((self.ndim, self.ndim),
                              dtype=theano.config.floatX),
            name='covmat',
            borrow=True
        )
        fun_update_covmat = theano.function(
            inputs=[data_batch],
            outputs=[],
            updates={covmat: covmat + T.dot(data_batch.T, data_batch) / np_ncases}
        )
        if verbose:
            print "Computing covariance, %d dots to punch:" % nbatches,
        for bidx in range(nbatches):
            if verbose:
                print ".",
                sys.stdout.flush()
            start = bidx * batch_size
            end = min((bidx + 1) * batch_size, ncases)
            fun_update_covmat(data[start:end, :])
        if verbose:     print "Done."

        # compute eigenvalue and eigenvector
        if verbose:     print "Eigen-decomposition...",; sys.stdout.flush()
        # u should be real valued vector, which stands for the variace of data
        # at each PC. v should be a real valued orthogonal matrix.
        u, v = numpy.linalg.eigh(covmat.get_value())
        v = v[:, numpy.argsort(u)[::-1]]
        u.sort()
        u = u[::-1]
        # throw away some eigenvalues for numerical stability
        self.stds = numpy.sqrt(u[u > 0.])
        self.std_fracs = self.stds.cumsum() / self.stds.sum()
        self.maxPCs = self.stds.shape[0]
        if verbose:     print "Done. Maximum stable PCs: %d" % self.maxPCs 
        
        # decide number of principle components.
        error_info = "Wrong \"retain\" value. Should be " + \
                     "a real number within the interval of (0, 1), " + \
                     "an integer in (0, maxPCs], None, or \'mle\'."
        if self.retain == None:
            self.retain = self.maxPCs
        elif self.retain == 'mle':
            raise NotImplementedError("Adaptive dimension matching," + \
                                      "not implemented yet...")
        elif isinstance(self.retain, int):
            assert (self.retain > 0 and self.retain <= self.maxPCs), error_info
        elif isinstance(self.retain, float):
            assert (self.retain > 0 and self.retain < 1), error_info
            self.retain = numpy.sum(self.std_fracs < self.retain) + 1
        if verbose:
            print "Number of selected PCs: %d, ratio of retained std: %f" % \
                (self.retain, self.std_fracs[self.retain-1])
        
        # decide if or not to whiten data
        if whiten:
            pca_forward = v[:, :self.retain] / self.stds[:self.retain]
            pca_backward = (v[:, :self.retain] * self.stds[:self.retain]).T
        else:
            pca_forward = v[:, :self.retain]
            pca_backward = pca_forward.T

        # build transforming layers
        pca_forward_w = theano.shared(
            value=pca_forward, name='pca_fwd', borrow=True
        )
        pca_forward_bvis = theano.shared(
            value = self.mean, name='pca_fwd_bvis', borrow=True
        )
        self.forward_layer = self.NeuralizedPCALayer(
            n_in=self.ndim, n_out=self.retain,
            init_w=pca_forward_w, init_bvis=pca_forward_bvis
        )

        pca_backward_w = theano.shared(
            value=pca_backward, name='pca_bkwd', borrow=True
        )
        pca_backward_bvis = theano.shared(
            value=self.mean, name='pca_bkwd_bvis', borrow=True
        )
        self.backward_layer = LinearLayer(
            n_in=self.retain, n_out=self.ndim,
            init_w=pca_backward_w, init_b=pca_backward_bvis
        )

    def forward(self, data, batch_size=10000, verbose=True):
        """
        Maps the given data to PCA representation, in a batchwise manner.
        
        There is no need to do the batchwise mapping though, but this
        implementation is for the unloaded version in the future. That will
        allow us to do PCA mapping on arbitrarilly large dataset.
        
        Parameters
        ------------
        data : numpy.ndarray 
            Data to be mapped.

        
        Returns
        ------------
        numpy.ndarray object.
        """
        assert hasattr(self, 'forward_layer'), 'You should fit PCA first.'
        data = data.astype(theano.config.floatX)
        ncases, ndim = data.shape
        assert ndim == self.ndim, \
            'Given data dimension doesn\'t match the learned PCA model.'
        nbatches = (ncases + batch_size - 1) / batch_size
        map_function = theano.function(
            [self.forward_layer.varin],
            self.forward_layer.output()
        )
        if verbose:
            print "Transforming, %d dots to punch:" % nbatches,
        pcaed_data = []
        for bidx in range(nbatches):
            if verbose:
                print ".",
                sys.stdout.flush()
            start = bidx * batch_size
            end = min((bidx + 1) * batch_size, ncases)
            pcaed_data.append(map_function(data[start:end, :]))
        pcaed_data = numpy.concatenate(pcaed_data, axis=0)
        if verbose:     print "Done."
        return pcaed_data

    def backward(self, data, batch_size=10000, verbose=True):
        """
        The same to forward(), but in a reverse direction.
        
        data : numpy.ndarray 
            Data to be mapped.

        Returns
        ------------
        numpy.ndarray object.
        """
        assert hasattr(self, 'backward_layer'), 'You should fit PCA first.'
        data = data.astype(theano.config.floatX)
        ncases, ndim = data.shape
        assert ndim == self.retain, \
            'Given data dimension doesn\'t match the learned PCA model.'
        nbatches = (ncases + batch_size - 1) / batch_size
        map_function = theano.function(
            [self.backward_layer.varin],
            self.backward_layer.output()
        )
        if verbose:
            print "Transforming, %d dots to punch:" % nbatches,
        recons_data = []
        for bidx in range(nbatches):
            if verbose:
                print ".",
                sys.stdout.flush()
            start = bidx * batch_size
            end = min((bidx + 1) * batch_size, ncases)
            recons_data.append(map_function(data[start:end, :]))
        recons_data = numpy.concatenate(recons_data, axis=0)
        if verbose:     print "Done."
        return recons_data

    def energy_dist(self,):
        """
        """
        assert hasattr(self, 'std_fracs'), "The PCA model has not fitted."
        return self.std_fracs


class SubtractMean(Layer):
    def __init__(self, n_in, varin=None):
        """
        For each sample, sbtract its mean value. So the output for one certain
        sample is fixed, and doesn't change w.r.t. other samples.
        """
        super(SubtractMean, self).__init__(n_in, n_in, varin=varin)

    def output(self):
        return (self.varin.T - self.varin.mean(axis=1)).T

    def _print_str(self):
        return "    (" + self.__class__.__name__ + ")"


class SubtractMeanAndNormalize(Layer):
    def __init__(self, n_in, varin=None):
        """
        This is also a sample-by-sample process. For each sample, subtract its
        mean value and then normalize values *within* the sample. So the output
        for one certain sample is also fixed, no matter what other samples are.
        """
        super(SubtractMeanAndNormalize, self).__init__(n_in, n_in, varin=varin)

    def output(self):
        mean_zero = (self.varin.T - self.varin.mean(axis=1)).T
        return (mean_zero.T / (mean_zero.std(axis=1) + 1e-10)).T
    
    def _print_str(self):
        return "    (" + self.__class__.__name__ + ")"
