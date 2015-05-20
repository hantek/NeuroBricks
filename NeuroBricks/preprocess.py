import sys
import copy
import numpy
import theano
import theano.tensor as T
from layer import Layer, LinearLayer

import pdb


class NeuralizedPCALayer(Layer):
    def __init__(self, n_in, n_out, init_w, init_bvis, varin=None):
        """
        The difference between a neuralized PCA Layer and a normal linear
        layer is the biases are on the visible side, and the weights are
        initialized as PCA transforming matrix.

        Though it is literally called PCA layer, but it is also used as
        building block for other transformations, like ZCA.
        """
        super(NeuralizedPCALayer, self).__init__(n_in, n_out, varin=varin)

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


class PCA(object):
    """
    A theano based PCA capable of using GPU.
    """
    """
    considering to make PCA a layer object

    def __init__(self, n_in, n_out, varin=None):
        pca_forward_w = theano.shared(
            value=pca_forward, name='pca_fwd', borrow=True
        )
        pca_forward_bvis = theano.shared(
            value = self.mean, name='pca_fwd_bvis', borrow=True
        )
        self.forward_layer = NeuralizedPCALayer(
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
    """

    def fit(self, data, retain=None, verbose=True, whiten=False):
        """
        Part of the code is adapted from Roland Memisevic's code.

        fit() deals with small datasets, i.e., those datasets that can be
        loaded into memory at once. It establishes 2 LinearLayer objects:
        PCAForwardLayer and PCABackwardLayer. They define how the data is
        mapped after the PCA mapping is learned.
        """
        self.retain = retain
        assert isinstance(data, numpy.ndarray), \
               "data has to be a numpy ndarray."
        data = data.copy().astype(theano.config.floatX)
        ncases, self.ndim = data.shape
        
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
        if verbose:
            print "Centralizing data..."
        data_variable = T.matrix('data_variable')
        np_ncases = numpy.array([ncases]).astype(theano.config.floatX)
        fun_partmean = theano.function(
            inputs=[data_variable],
            outputs=T.sum(data_variable / np_ncases, axis=0)
        )
        self.mean = numpy.zeros(self.ndim, dtype=theano.config.floatX)
        self.mean += fun_partmean(data)
        data -= self.mean
        if verbose:     print "Done."
        
        # compute convariance matrix
        if verbose:
            print "Computing covariance..."
        covmat = theano.shared(
            value=numpy.zeros((self.ndim, self.ndim),
                              dtype=theano.config.floatX),
            name='covmat',
            borrow=True
        )
        fun_update_covmat = theano.function(
            inputs=[data_variable],
            outputs=[],
            updates={covmat: covmat + \
                             T.dot(data_variable.T, data_variable) / np_ncases}
        )
        fun_update_covmat(data)
        self.covmat = covmat.get_value()
        if verbose:     print "Done."

        # compute eigenvalue and eigenvector
        if verbose:     print "Eigen-decomposition...",; sys.stdout.flush()
        # u should be real valued vector, which stands for the variace of data
        # at each PC. v should be a real valued orthogonal matrix.
        u, v_unsorted = numpy.linalg.eigh(self.covmat)
        self.v = v_unsorted[:, numpy.argsort(u)[::-1]]
        u.sort()
        u = u[::-1]
        # throw away some eigenvalues for numerical stability
        self.stds = numpy.sqrt(u[u > 0.])
        self.variance_fracs = (self.stds ** 2).cumsum() / (self.stds ** 2).sum()
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
            self.retain = numpy.sum(self.variance_fracs < self.retain) + 1
        if verbose:
            print "Number of selected PCs: %d, ratio of retained variance: %f"%\
                (self.retain, self.variance_fracs[self.retain-1])
        self._build_layers(whiten)
   
    def fit_partwise(self, data_genfun, data_resetfun, ncases, ndim,
                     retain=None, verbose=True, whiten=False):
        """
        fit_partwise() is for computing PCA for large datasets. the data part
        is generated by a generator, and at each iteration the generated data
        should be in the form of a single numpy.ndarray, with 2-d structure. 
        The method establishes 2 LinearLayer objects: PCAForwardLayer and
        PCABackwardLayer. They define how the data is mapped after the PCA
        mapping is learned.
        """
        self.retain = retain
        self.ndim = ndim

        # centralizing data
        if verbose:
            print "Centralizing data..."
        data_variable = T.matrix('data_variable')
        np_ncases = numpy.array([ncases]).astype(theano.config.floatX)
        fun_partmean = theano.function(
            inputs=[data_variable],
            outputs=T.sum(data_variable / np_ncases, axis=0)
        )
        self.mean = numpy.zeros(self.ndim, dtype=theano.config.floatX)
        data_resetfun()
        data_generator = data_genfun()
        for data_part in data_generator:
            assert isinstance(data_part, numpy.ndarray), (
                "data_genfun has to be a generator function yielding "
                "numpy.ndarray.")
            data_part = data_part.astype(theano.config.floatX)
            _, self.ndim = data_part.shape
            self.mean += fun_partmean(data_part)
            if verbose:
                print ".",
                sys.stdout.flush()
        if verbose:     print "Done."
        
        # compute convariance matrix
        if verbose:
            print "Computing covariance..."
        covmat = theano.shared(
            value=numpy.zeros((self.ndim, self.ndim),
                              dtype=theano.config.floatX),
            name='covmat',
            borrow=True
        )
        fun_update_covmat = theano.function(
            inputs=[data_variable],
            outputs=[],
            updates={covmat: covmat + \
                             T.dot(data_variable.T, data_variable) / np_ncases}
        )
        data_resetfun()
        data_generator = data_genfun()
        for data_part in data_generator:
            data_part = data_part.astype(theano.config.floatX) - self.mean
            fun_update_covmat(data_part)
            if verbose:
                print ".",
                sys.stdout.flush()
        self.covmat = covmat.get_value()
        if verbose:     print "Done."

        # compute eigenvalue and eigenvector
        if verbose:     print "Eigen-decomposition...",; sys.stdout.flush()
        # u should be real valued vector, which stands for the variace of data
        # at each PC. v should be a real valued orthogonal matrix.
        u, v_unsorted = numpy.linalg.eigh(self.covmat)
        self.v = v_unsorted[:, numpy.argsort(u)[::-1]]
        u.sort()
        u = u[::-1]
        # throw away some eigenvalues for numerical stability
        self.stds = numpy.sqrt(u[u > 0.])
        self.variance_fracs = (self.stds ** 2).cumsum() / (self.stds ** 2).sum()
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
            self.retain = numpy.sum(self.variance_fracs < self.retain) + 1
        if verbose:
            print "Number of selected PCs: %d, ratio of retained variance: %f"%\
                (self.retain, self.variance_fracs[self.retain-1])
        self._build_layers(whiten)
 
    def _build_layers(self, whiten):        
        # decide if or not to whiten data
        if whiten:
            pca_forward = self.v[:, :self.retain] / self.stds[:self.retain]
            pca_backward = (self.v[:, :self.retain] * self.stds[:self.retain]).T
        else:
            pca_forward = self.v[:, :self.retain]
            pca_backward = pca_forward.T

        # build transforming layers
        pca_forward_w = theano.shared(
            value=pca_forward, name='pca_fwd', borrow=True
        )
        pca_forward_bvis = theano.shared(
            value = self.mean, name='pca_fwd_bvis', borrow=True
        )
        self.forward_layer = NeuralizedPCALayer(
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
        self.outdim = self.retain

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
        assert hasattr(self, 'forward_layer'), 'Please fit the model first.'
        data = data.astype(theano.config.floatX)
        ncases, ndim = data.shape
        assert ndim == self.ndim, \
            'Given data dimension doesn\'t match the learned model.'
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
        assert hasattr(self, 'backward_layer'), 'Please fit the model first.'
        data = data.astype(theano.config.floatX)
        ncases, ndim = data.shape
        assert ndim == self.outdim, \
            'Given data dimension doesn\'t match the learned model.'
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
        assert hasattr(self, 'variance_fracs'), \
            "The model has not been fitted."
        return self.variance_fracs


class ZCA(PCA):
    def _build_layers(self, whiten):
        # decide if or not to whiten data
        if whiten:
            zca_forward = numpy.dot(
                self.v[:, :self.retain] / self.stds[:self.retain],
                self.v[:, :self.retain].T
            )
            zca_backward = numpy.dot(
                self.v[:, :self.retain],
                (self.v[:, :self.retain] * self.stds[:self.retain]).T
            )
        else:
            zca_forward = numpy.dot(
                self.v[:, :self.retain],
                self.v[:, :self.retain].T
            )
            zca_backward = zca_forward
            
        # build transforming layers
        zca_forward_w = theano.shared(
            value=zca_forward, name='zca_fwd', borrow=True
        )
        zca_forward_bvis = theano.shared(
            value=self.mean, name='zca_fwd_bvis', borrow=True
        )
        self.forward_layer = NeuralizedPCALayer(
            n_in=self.ndim, n_out=self.ndim,
            init_w=zca_forward_w, init_bvis=zca_forward_bvis
        )

        zca_backward_w = theano.shared(
            value=zca_backward, name='zca_bkwd', borrow=True
        )
        zca_backward_bvis = theano.shared(
            value=self.mean, name='zca_bkwd_bvis', borrow=True
        )
        self.backward_layer = LinearLayer(
            n_in=self.ndim, n_out=self.ndim,
            init_w=zca_backward_w, init_b=zca_backward_bvis
        )
        self.outdim = self.ndim


class SubtractMean(Layer):
    def __init__(self, n_in, varin=None):
        """
        For each sample, subtract its mean value. So the output for one certain
        sample is fixed, and doesn't change w.r.t. other samples.
        """
        super(SubtractMean, self).__init__(n_in, n_in, varin=varin)

    def output(self):
        return (self.varin.T - self.varin.mean(axis=1)).T

    def _print_str(self):
        return "    (" + self.__class__.__name__ + ")"


class SubtractMeanAndNormalizeH(Layer):
    def __init__(self, n_in, varin=None):
        """
        This is also a sample-by-sample process. For each sample, subtract its
        mean value and then normalize values *within* the sample. So the output
        for one certain sample is also fixed, no matter what other samples are.
        """
        super(SubtractMeanAndNormalizeH, self).__init__(n_in, n_in, varin=varin)

    def output(self):
        mean_zero = (self.varin.T - self.varin.mean(axis=1)).T
        return (mean_zero.T / (mean_zero.std(axis=1) + 1e-10)).T
    
    def _print_str(self):
        return "    (" + self.__class__.__name__ + ")"


class SubtractMeanAndNormalize(SubtractMeanAndNormalizeH):
    def __init__(self, n_in, varin=None):
        super(SubtractMeanAndNormalize, self).__init__(n_in, varin=varin)
        print ("\nWarning: SubtractMeanAndNormalize is depreciated, use "
               "SubtractMeanAndNormalizeH instead.\n")


class SubtractMeanAndNormalizeV(Layer):
    def __init__(self, n_in, varin=None):
        """
        This is a process which normalizes each dimension across the whole
        dataset. So before applying it, this layer itself needs to be fitted.
        """
        super(SubtractMeanAndNormalizeV, self).__init__(n_in, n_in, varin=varin)
        self.bias = theano.shared(
            value=numpy.zeros(n_in, dtype=theano.config.floatX),
            name='bias_meanormV',
            borrow=True)
        self.scale = theano.shared(
            value=numpy.zeros(n_in, dtype=theano.config.floatX),
            name='scale_meanormV',
            borrow=True)

    def fit(self, data):
        self.bias.set_value(-data.mean(axis=0))
        self.scale.set_value(data.std(axis=0))

    def output(self):
        return (self.varin + self.bias) / (self.scale + 1e-10)
    
    def _print_str(self):
        return "    (" + self.__class__.__name__ + ")"


