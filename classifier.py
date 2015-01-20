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
        super(LogisticRegression, self).__init__(
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
        """The output of a logistic regressor is p_y_given_x."""
        return T.nnet.softmax(self.fanin())

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

    def weightdecay_cost(self, weightdecay=1e-3):
        return self.cost() + weightdecay * (self.w**2).sum()

    def predict(self):
        return T.argmax(self.output(), axis=1)


class Perceptron(Classifier):
    def __init__(self):
        #
        # TODO:
        #
        pass


class LinearRegression(Classifier):
    def __init__(self):
        #
        # TODO:
        #
        pass
