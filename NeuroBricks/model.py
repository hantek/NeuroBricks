"""
This file defines some basic models. A default cost is defined here. By overidi-
ng it subclasses can have more complex forms of costs.

Nominators with "_get_" stands for they are dealing with numpy arrays.
"""
import numpy
import theano
import theano.tensor as T
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from layer import Layer, SigmoidLayer, LinearLayer, ZerobiasLayer, ReluLayer

import pdb

class Model(Layer):
    """
    A base class for models, which by inheritation can turn to be a concreate
    classifier or regressor.
    """
    def __init__(self, n_in, n_out, varin=None):
        super(Model, self).__init__(n_in, n_out, varin=varin)

    def fanin(self):
        print "Calling the fan-in of a base model might have no meaning."

    def output(self):
        print "Calling the output of a base model might have no meaning."

    def activ_prime(self):
        print "Calling the derivative of activation of a base model might " + \
              "have no meaning."
    
    def cost(self):
        raise NotImplementedError("Must be implemented by subclass.")


class AutoEncoder(Model):
    def __init__(self, n_in, n_hid, vistype, varin=None):
        """
        It's better to construct an autoencoder by constructing an instance of a
        certain autoencoder class instead of using "+" between two already
        instanciated encoder and decoder. Because that helps dealing with a lot
        of parameter settings and guarantees a bunch of autoencoder-specific
        methods, along with analyzing & visualizing methods.
        """
        super(AutoEncoder, self).__init__(n_in, n_hid, varin=varin)
        self.n_hid = self.n_out

        # Note that we define this self.params_private member data to stand for
        # the "private" parameters for the autoencoder itself. i.e., parameters
        # in BOTH the encoder and decoder. This is ONLY for layerwise training
        # the autoencoder itself  while it were stacked into a deep structure.
        self.params_private = self.params + []

        # While, for self.params, we only include parameters of the encoder.
        # Defining self.params and self.params_private in this way helps
        # collecting "params" member data (which is for training the whole
        # network in the final fine-tuning stage.) while stacking layers.

        # for plotting
        self.vistype = vistype
        if vistype == 'binary':
            self.rangex=[0., 1.]
            self.rangey=[0., 1.]
        elif vistype == 'real':
            self.rangex=[-1., 1.]
            self.rangey=[-1., 1.]
        else:
            raise ValueError("vistype has to be either 'binary' or 'real'.")

    def encoder(self):
        """
        We must ensure there is *NO* parameters passed to the encoder() method.
        All initialization parameters should be set in the instantiation of
        model. That ensures we are not invoking different models between
        different encoder() calls.

        The autoencoder ADT defines a 2-layer structure (both layer might be
        a StackedLayer object, in that way it will have more than 2 actual
        layers) which allows for reconstructing the input from middle hidden
        layers.
        """
        raise NotImplementedError("Must be implemented by subclass.")

    def decoder(self):
        raise NotImplementedError("Must be implemented by subclass.")

    def fanin(self):
        """
        The fanin of the first actual layer might still have sense. so keep it
        here.
        """
        return self.encoder().fanin()

    def hidden(self):
        return self.encoder().output()

    def output(self):
        """
        Setting the output in this manner helps stacking the encoders in a 
        convenient way. That allows you to get a StackedLayer object which only
        contains the encoder part in the returned deep structure while calling
            ... + Autoencoder(...) + Layer(...) + ...
        """
        return self.hidden()

    def activ_prime(self):
        return self.encoder().activ_prime()
    
    def reconstruction(self):
        return self.decoder().output()

    def cost(self):
        """
        By default it presents a cross entropy cost for binary data and MSE for
        real valued data. We can still overide it if we were using a different 
        training criteria.
        """
        x_head = self.reconstruction()
        if self.vistype == 'binary':
            cost_per_case = - T.sum(self.varin * T.log(x_head) + \
                (1 - self.varin) * T.log(1 - x_head), axis=1)
            return T.mean(cost_per_case)
        elif self.vistype == 'real':
            cost_per_case = T.sum((x_head - self.varin) ** 2, axis=1)
            return T.mean(cost_per_case)
        else:
            raise ValueError("vistype has to be either binary or real.")


    # Following are for analysis -----------------------------------------------

    def score(self, fig_handle=None):
        assert self.n_in == 2, "Cannot draw score for higher dimentions."
        #
        # TODO: plot the score of autoencoder for a 2-D input, if fig_handle 
        # given, plot on the given figure.
        #
        pass

    def quiver(self, rangex=None, rangey=None, dpi=50, verbose=True,
               filename='default_draw_aequiver.png'): 
        assert self.n_in == 2, "Cannot draw quiver for higher dimentions."
        if rangex != None:    self.rangex = rangex
        if rangey != None:    self.rangey = rangey
        if not hasattr(self, '_fig_quiver'):  # first call
            self._fig_quiver = plt.figure()
            self.ax = self._fig_quiver.add_subplot(111)
            self._get_reconstruction = theano.function(
                [self.encoder().varin], self.reconstruction())
            
            np_x, np_y = numpy.meshgrid(
                numpy.linspace(self.rangex[0], self.rangex[1], num=dpi),
                numpy.linspace(self.rangey[0], self.rangey[1], num=dpi))
            self.mesh_points = numpy.concatenate(
                (np_x[:, :, numpy.newaxis], np_y[:, :, numpy.newaxis]), 
                axis=2).reshape(-1, 2)
            
            recons = self._get_reconstruction(self.mesh_points)
            diff = recons - self.mesh_points
            quiver_x = diff[:, 0].reshape(dpi, dpi)
            quiver_y = diff[:, 1].reshape(dpi, dpi)

            self.Q = self.ax.quiver(quiver_x, quiver_y)
            self._fig_quiver.canvas.draw()

        else:
            recons = self._get_reconstruction(self.mesh_points)
            diff = recons - self.mesh_points
            quiver_x = diff[:, 0].reshape(dpi, dpi)
            quiver_y = diff[:, 1].reshape(dpi, dpi)
            
            self.Q.set_UVC(quiver_x, quiver_y)
            self._fig_quiver.canvas.draw()

        if verbose:
            plt.pause(0.0001)
        else:
            plt.savefig(filename)


class ClassicalAutoencoder(AutoEncoder):
    def __init__(self, n_in, n_hid, vistype, hidtype=None, varin=None, tie=True,
                 init_w=None, init_b=None, init_wT=None, init_bT=None,
                 npy_rng=None):
        super(ClassicalAutoencoder, self).__init__(
            n_in, n_hid, vistype=vistype, varin=varin
        )

        if not hidtype:
            hidtype = self.vistype
        assert (hidtype == 'binary' or hidtype == 'real')
        self.hidtype = hidtype

        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)

        if not init_w:
            w = numpy.asarray(npy_rng.uniform(
                low=-4 * numpy.sqrt(6. / (self.n_in + self.n_hid)),
                high=4 * numpy.sqrt(6. / (self.n_in + self.n_hid)),
                size=(self.n_in, self.n_hid)), dtype=theano.config.floatX)
            init_w = theano.shared(value=w, name='w_ae', borrow=True)
        # else:
        #     TODO. The following assetion is complaining about an attribute
        #     error while passing w.T to init_w. Considering using a more
        #     robust way of assertion in the future.
        #     assert init_w.get_value().shape == (n_in, n_out)
        self.w = init_w

        if tie:
            assert init_wT == None, "Tied autoencoder do not accept init_wT."
            init_wT = self.w.T
        else:
            if not init_wT:
                wT = numpy.asarray(npy_rng.uniform(
                    low = -4 * numpy.sqrt(6. / (self.n_hid + self.n_in)),
                    high = 4 * numpy.sqrt(6. / (self.n_hid + self.n_in)),
                    size=(self.n_hid, self.n_in)), dtype=theano.config.floatX)
                init_wT = theano.shared(value=w, name='wT_ae', borrow=True)
            # else:
            #     TODO. The following assetion is complaining about an attribute
            #     error while passing w.T to init_w. Considering using a more
            #     robust way of assertion in the future.
            #     assert init_wT.get_value().shape == (n_in, n_out)
        self.wT = init_wT

        if not init_b:
            init_b = theano.shared(value=numpy.zeros(
                                       self.n_hid,
                                       dtype=theano.config.floatX),
                                   name='b_ae_encoder', borrow=True)
        else:
            assert init_b.get_value().shape == (self.n_hid,)
        self.b = init_b
        
        if not init_bT:
            init_bT = theano.shared(value=numpy.zeros(
                                        self.n_in,
                                        dtype=theano.config.floatX),
                                    name='b_ae_decoder', borrow=True)
        else:
            assert init_bT.get_value().shape == (self.n_in,)
        self.bT = init_bT

        self.params = [self.w, self.b]
        if tie:
            self.params_private = self.params + [self.bT]
        else:
            self.params_private = self.params + [self.wT, self.bT]

    def encoder(self):
        if self.hidtype == 'binary':
            return SigmoidLayer(
                self.n_in, self.n_hid, varin=self.varin,
                init_w=self.w, init_b=self.b
            )
        elif self.hidtype == 'real':
            return LinearLayer(
                self.n_in, self.n_hid, varin=self.varin,
                init_w=self.w, init_b=self.b
            )
        
    def decoder(self):
        if self.vistype == 'binary':
            return SigmoidLayer(
                self.n_hid, self.n_in, varin=self.encoder().output(),
                init_w=self.wT, init_b=self.bT
            )
        elif self.vistype == 'real':
            return LinearLayer(
                self.n_hid, self.n_in, varin=self.encoder().output(),
                init_w=self.wT, init_b=self.bT
            )

    def contraction(self, contract_level=0.1):
        """
        We are computing the Jacobian, i.e. dh / dv, implicitly. The equivalent
        explicit way is:
            jacobian = dh.dimshuffle(0, 1, 'x') * self.w.dimshuffle('x', 1, 0)
            contraction_per_case = T.sum(jacobian ** 2, axis=(1, 2))
        This implementation is inspired by https://groups.google.com/forum/#!to-
        pic/pylearn-dev/iY7swxgn-xI/discussion . It helps speeding up
        computation and reducing memory usage dramatically.
        """
        if self.hidtype == 'binary':
            dh = self.hidden() * (1 - self.hidden())  # sigmoid derivative
        elif self.hidtype == 'real':
            dh = 1.  # self.hidden()  # linear activation derivative
        contraction_per_case = T.sum(T.dot(dh**2, self.w.T**2), axis=1)
        return contract_level * T.mean(contraction_per_case)


class ZerobiasAutoencoder(AutoEncoder):
    def __init__(self, n_in, n_hid, vistype, threshold=1.0, varin=None,
                 tie=True, init_w=None, init_wT=None, init_bT=None, 
                 npy_rng=None):
        super(ZerobiasAutoencoder, self).__init__(
            n_in, n_hid, vistype=vistype, varin=varin
        )
        self.threshold = threshold

        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)

        if not init_w:
            w = numpy.asarray(npy_rng.uniform(
                low=-4 * numpy.sqrt(6. / (self.n_in + self.n_hid)),
                high=4 * numpy.sqrt(6. / (self.n_in + self.n_hid)),
                size=(self.n_in, self.n_hid)), dtype=theano.config.floatX)
            init_w = theano.shared(value=w, name='w_zae', borrow=True)
        # else:
        #     TODO. The following assetion is complaining about an attribute
        #     error while passing w.T to init_w. Considering using a more
        #     robust way of assertion in the future.
        #     assert init_w.get_value().shape == (n_in, n_out)
        self.w = init_w

        if tie:
            assert init_wT == None, "Tied autoencoder do not accept init_wT."
            init_wT = self.w.T
        else:
            if not init_wT:
                wT = numpy.asarray(npy_rng.uniform(
                    low = -4 * numpy.sqrt(6. / (self.n_hid + self.n_in)),
                    high = 4 * numpy.sqrt(6. / (self.n_hid + self.n_in)),
                    size=(self.n_hid, self.n_in)), dtype=theano.config.floatX)
                init_wT = theano.shared(value=w, name='wT_zae', borrow=True)
            # else:
            #     TODO. The following assetion is complaining about an attribute
            #     error while passing w.T to init_w. Considering using a more
            #     robust way of assertion in the future.
            #     assert init_wT.get_value().shape == (n_in, n_out)
        self.wT = init_wT

        if not init_bT:
            init_bT = theano.shared(value=numpy.zeros(
                                        self.n_in, dtype=theano.config.floatX),
                                    name='b_zae_decoder', borrow=True)
        else:
            assert init_bT.get_value().shape == (self.n_in,)
        self.bT = init_bT

        self.params = [self.w]
        if tie:
            self.params_private = self.params + [self.bT]
        else:
            self.params_private = self.params + [self.wT, self.bT]

    def encoder(self):
        return ZerobiasLayer(
            self.n_in, self.n_hid, threshold=self.threshold,
            varin=self.varin, init_w=self.w
        )

    def decoder(self):
        if self.vistype == 'binary':
            return SigmoidLayer(
                self.n_hid, self.n_in, varin=self.encoder().output(),
                init_w=self.wT, init_b=self.bT
            )
        elif self.vistype == 'real':
            return LinearLayer(
                self.n_hid, self.n_in, varin=self.encoder().output(),
                init_w=self.wT, init_b=self.bT
            )


class ReluAutoencoder(AutoEncoder):
    def __init__(self, n_in, n_hid, vistype, varin=None, tie=True,
                 init_w=None, init_b=None, init_wT=None, init_bT=None, 
                 npy_rng=None):
        super(ReluAutoencoder, self).__init__(
            n_in, n_hid, vistype=vistype, varin=varin
        )

        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)

        if not init_w:
            w = numpy.asarray(npy_rng.uniform(
                low=-4 * numpy.sqrt(6. / (self.n_in + self.n_hid)),
                high=4 * numpy.sqrt(6. / (self.n_in + self.n_hid)),
                size=(self.n_in, self.n_hid)), dtype=theano.config.floatX)
            init_w = theano.shared(value=w, name='w_reluae', borrow=True)
        # else:
        #     TODO. The following assetion is complaining about an attribute
        #     error while passing w.T to init_w. Considering using a more
        #     robust way of assertion in the future.
        #     assert init_w.get_value().shape == (n_in, n_out)
        self.w = init_w

        if not init_b:
            init_b = theano.shared(
                value=numpy.zeros(self.n_hid, dtype=theano.config.floatX),
                name='b_reluae_encoder',
                borrow=True
            )
        else:
            assert init_b.get_value().shape == (self.n_hid,)
        self.b = init_b
        
        if tie:
            assert init_wT == None, "Tied autoencoder do not accept init_wT."
            init_wT = self.w.T
        else:
            if not init_wT:
                wT = numpy.asarray(npy_rng.uniform(
                    low = -4 * numpy.sqrt(6. / (self.n_hid + self.n_in)),
                    high = 4 * numpy.sqrt(6. / (self.n_hid + self.n_in)),
                    size=(self.n_hid, self.n_in)), dtype=theano.config.floatX)
                init_wT = theano.shared(value=w, name='wT_reluae', borrow=True)
            # else:
            #     TODO. The following assetion is complaining about an attribute
            #     error while passing w.T to init_w. Considering using a more
            #     robust way of assertion in the future.
            #     assert init_wT.get_value().shape == (n_in, n_out)
        self.wT = init_wT

        if not init_bT:
            init_bT = theano.shared(
                value=numpy.zeros(self.n_in, dtype=theano.config.floatX),
                name='b_reluae_decoder',
                borrow=True
            )
        else:
            assert init_bT.get_value().shape == (self.n_in,)
        self.bT = init_bT

        self.params = [self.w, self.b]
        if tie:
            self.params_private = self.params + [self.bT]
        else:
            self.params_private = self.params + [self.wT, self.bT]

    def encoder(self):
        return ReluLayer(
            self.n_in, self.n_hid,
            varin=self.varin, init_w=self.w, init_b=self.b
        )

    def decoder(self):
        if self.vistype == 'binary':
            return SigmoidLayer(
                self.n_hid, self.n_in, varin=self.encoder().output(),
                init_w=self.wT, init_b=self.bT
            )
        elif self.vistype == 'real':
            return LinearLayer(
                self.n_hid, self.n_in, varin=self.encoder().output(),
                init_w=self.wT, init_b=self.bT
            )
