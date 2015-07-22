"""
Here are some classifiers.
"""
import numpy
import theano
import theano.tensor as T
from sklearn.metrics import confusion_matrix
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import Model
from layer import SigmoidLayer


class Classifier(Model):
    def __init__(self, n_in, n_out, varin=None, vartruth=None):
        super(Classifier, self).__init__(n_in, n_out, varin=varin)
        if not vartruth:
            vartruth = T.lvector('truth')
        assert isinstance(vartruth, T.TensorVariable)
        self.vartruth = vartruth

    def output(self):
        """
        output() is defined as the activity of the highest layer, not the 
        prediction results. Prediction results are generated in the predict()
        method. The input and output should always be theano variables.
        """
        raise NotImplementedError("Must be implemented by subclass.")

    def predict(self):
        raise NotImplementedError("Must be implemented by subclass.")

    # Following are for analysis ----------------------------------------------

    def analyze_performance(self, data, truth, verbose=True):
        """
        TODO: WRITEME

        data : numpy.ndarray
        truth : numpy.ndarray
        verbose : bool
        """
        assert data.shape[0] == truth.shape[0], "Data and truth shape mismatch."
        if not hasattr(self, '_get_predict'):
            self._get_predict = theano.function([self.varin], self.predict())

        cm = confusion_matrix(truth, self.get_predict(data))
        pr_a = cm.trace()*1.0 / test_truth.size
        pr_e = ((cm.sum(axis=0)*1.0 / test_truth.size) * \
            (cm.sum(axis=1)*1.0 / test_truth.size)).sum()
        k = (pr_a - pr_e) / (1 - pr_e)
        print "OA: %f, kappa index of agreement: %f" % (pr_a, k)
        if verbose: # Show confusion matrix
            if not hasattr(self, '_fig_confusion'):
                self._fig_confusion = plt.gcf()
                self.ax = self._fig_confusion.add_subplot(111)
                self.confmtx = self.ax.matshow(cm)
                plt.title("Confusion Matrix")
                self._fig_confusion.canvas.show()
            else:
                self.confmtx.set_data(cm)
                self._fig_confusion.canvas.draw()
            print "confusion matrix:"
            print cm


class LogisticRegression(Classifier):
    def __init__(self, n_in, n_out, varin=None, vartruth=None, 
                 init_w=None, init_b=None, npy_rng=None):
        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)
        self.npy_rng = npy_rng
        
        self.n_out = n_out
        if not init_b:
            init_b = theano.shared(
                value=numpy.zeros(n_out, dtype=theano.config.floatX),
                name=self.__class__.__name__ + '_b', borrow=True)
        else:
            assert init_b.get_value().shape == (n_out,)
        self.b = init_b

        self.vartruth = vartruth
        self.init_w = init_w
        self.n_in, self.varin = n_in, varin
        if self.n_in != None:
            self._init_complete()

    def _init_complete(self):
        assert self.n_in != None, "Need to have n_in attribute to execute."
        super(LogisticRegression, self).__init__(
            self.n_in, self.n_out, varin=self.varin, vartruth=self.vartruth
        )

        if not self.init_w:
            w = numpy.asarray(self.npy_rng.uniform(
                low = -4 * numpy.sqrt(6. / (self.n_in + self.n_out)),
                high = 4 * numpy.sqrt(6. / (self.n_in + self.n_out)),
                size=(self.n_in, self.n_out)), dtype=theano.config.floatX)
            self.init_w = theano.shared(value=w,
                name=self.__class__.__name__ + '_w', borrow=True)
        self.w = self.init_w

        self.params = [self.w, self.b]

    def fanin(self):
        self.varfanin = T.dot(self.varin, self.w) + self.b
        return self.varfanin

    def output(self, fanin=None):
        """The output of a logistic regressor is p_y_given_x."""
        if fanin == None:   fanin = self.fanin()
        return T.nnet.softmax(fanin)

    def activ_prime(self):
        """for the special relationship between softmax and sigmoid, we can
        define this method in this classifier."""
        return SigmoidLayer(
            self.n_in, self.n_out, varin=self.varin,
            init_w=self.w, init_b=self.b
        ).activ_prime()

    def cost(self):
        return -T.mean(
            T.log(self.output())[
                T.arange(self.vartruth.shape[0]), self.vartruth]
        )

    def weightdecay(self, weightdecay=1e-3):
        return weightdecay * (self.w**2).sum()

    def predict(self):
        return T.argmax(self.output(), axis=1)


class BinaryLogisticRegression(LogisticRegression):
    def __init__(self, n_in, n_out, mode='stochastic', varin=None,
                 vartruth=None, init_w=None, init_b=None, npy_rng=None):
        super(BinaryLogisticRegression, self).__init__(
            n_in=n_in, n_out=n_out, varin=varin, vartruth=vartruth,
            init_w=init_w, init_b=init_b, npy_rng=npy_rng
        )
        self.srng = theano.sandbox.rng_mrg.MRG_RandomStreams(
            self.npy_rng.randint(999999))
        self.mode = mode
        
    def fanin(self):
        self.varfanin = T.dot(self.varin, self.binarized_weight()) + self.b
        return self.varfanin
    
    def binarized_weight(self):
        self.w0 = (
            4 * numpy.sqrt(6. / (self.n_in + self.n_out)) / 2
        ).astype(theano.config.floatX)
        if self.mode == 'deterministic':
            self.wb = T.switch(T.ge(self.w, 0), self.w0, -self.w0)

        elif self.mode == 'stochastic':
            # probability=hard_sigmoid(w/w0)
            p = T.clip(((self.w / self.w0) + 1) / 2, 0, 1)
            p_mask = T.cast(self.srng.binomial(n=1, p=p, size=T.shape(self.w)),
                            theano.config.floatX)

            # [0,1] -> -W0 or W0
            self.wb = T.switch(p_mask, self.w0, -self.w0)

        else:
            raise ValueError("Parameter 'self.mode' has to be either "
                             "'deterministic' or 'stochastic'")

        return self.wb

    def quantized_bprop(self, cost):
        index_low = T.switch(self.varin > 0.,
            T.floor(T.log2(self.varin)), T.floor(T.log2(-self.varin))
        )
        index_low = T.clip(index_low, -4, 3)
        sign = T.switch(self.varin > 0., 1., -1.)
        # the upper 2**(integer power) though not used explicitly.
        # index_up = index_low + 1
        # percentage of upper index.
        p_up = sign * self.varin / 2**(index_low) - 1
        index_random = index_low + self.srng.binomial(
            n=1, p=p_up, size=T.shape(self.varin), dtype=theano.config.floatX)
        # quantized_rep = sign * 2**index_random
        quantized_rep = self.varin

        error = T.grad(cost=cost, wrt=self.varfanin)

        self.dEdW = T.dot(quantized_rep.T, error)


class LinearSVM(LogisticRegression):
    def output(self, fanin=None):
        """The output of a logistic regressor is p_y_given_x."""
        if fanin == None:   fanin = self.fanin()
        return fanin

    def activ_prime(self):
        raise NotImplementedError("Must be implemented by subclass.")

    def cost(self):
        """Squared hinge loss"""
        return T.mean(
            T.sqr(T.maximum(
                0.,
                1. - (2 * T.extra_ops.to_one_hot(
                    self.vartruth, self.n_out) - 1) * self.output()
            ))
        )


class BinaryLinearSVM(BinaryLogisticRegression):
    def output(self, fanin=None):
        """The output of a logistic regressor is p_y_given_x."""
        if fanin == None:   fanin = self.fanin()
        return fanin

    def activ_prime(self):
        raise NotImplementedError("Must be implemented by subclass.")

    def cost(self):
        """Squared hinge loss"""
        return T.mean(
            T.sqr(T.maximum(
                0.,
                1. - (2 * T.extra_ops.to_one_hot(
                    self.vartruth, self.n_out) - 1) * self.output()
            ))
        )


class Perceptron(Classifier):
    def __init__(self):
        #
        # TODO:
        #
        pass


class LinearRegression(Classifier):
    def __init__(self, n_in, n_out, varin=None, vartruth=None, 
                 init_w=None, init_b=None, npy_rng=None):
        super(LinearRegression, self).__init__(
            n_in, n_out, varin=varin, vartruth=vartruth
        )

        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)
        self.npy_rng = npy_rng

        if not init_w:
            w = numpy.asarray(npy_rng.uniform(
                low = -4 * numpy.sqrt(6. / (n_in + n_out)),
                high = 4 * numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            init_w = theano.shared(value=w, name='w_sigmoid', borrow=True)
        # else:
        #     TODO. The following assetion is complaining about an attribute
        #     error while passing w.T to init_w. Considering using a more
        #     robust way of assertion in the future.
        #     assert init_w.get_value().shape == (n_in, n_out)
        self.w = init_w

        if not init_b:
            init_b = theano.shared(value=numpy.zeros(
                                       n_out,
                                       dtype=theano.config.floatX),
                                   name='b_sigmoid', borrow=True)
        else:
            assert init_b.get_value().shape == (n_out,)
        self.b = init_b

        self.params = [self.w, self.b]

    def fanin(self):
        return T.dot(self.varin, self.w) + self.b

    def output(self):
        return self.fanin()

    def activ_prime(self):
        """for the special relationship between LinearRegression and LinearLayer, we can
        define this method in this classifier."""
        return LinearLayer(
            self.n_in, self.n_out, varin=self.varin,
            init_w=self.w, init_b=self.b
        ).activ_prime()

    def cost(self):
        # TODO not known
        raise NotImplementedError("ERROR")
        return str("ERROR")
    
    def weightdecay(self, weightdecay=1e-3):
        return weightdecay * (self.w**2).sum()

    def predict(self):
        return T.argmax(self.output(), axis=1)
