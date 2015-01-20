"""
All parameters (excluding superparameters) in the model should be in theano var-
iables or theano shared values. In the training part, these variables should be
organized into "theano.function"s. So there should be no theano.function in the 
definition of models here.
"""
import numpy
import theano
import theano.tensor as T
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb


class Layer(object):
    def __init__(self, n_in, n_out, varin=None):
        """
        Parameters
        -----------
        n_in : int
        n_out : int
        varin : theano.tensor.TensorVariable, optional
        """
        self.n_in = n_in
        self.n_out = n_out

        if not varin:
            varin = T.matrix('varin')
        assert isinstance(varin, T.TensorVariable)
        self.varin = varin

        self.params = []  # to be implemented by subclass

    def fanin(self):
        raise NotImplementedError("Must be implemented by subclass.")

    def output(self):
        raise NotImplementedError("Must be implemented by subclass.")

    def activ_prime(self):
        """
        Value of derivative of the activation function w.r.t fanin(), given 
        fanin() as the argument of the derivative funciton.
        """
        raise NotImplementedError("Must be implemented by subclass.")

    def _print_str(self):
        return self.__class__.__name__ + ": " + str(self.n_in) + " --> " \
                                              + str(self.n_out)

    def print_layer(self):
        print self._print_str()

    def __add__(self, other):
        """It is used for conveniently construct stacked layers."""
        assert isinstance(other, Layer), "Addition not defined."
        if hasattr(self, 'models_stack'):
            models_left = self.models_stack
        else: 
            models_left = [self]

        if hasattr(other, 'models_stack'):
            models_right = other.models_stack
        else: 
            models_right = [other]

        models_stack = models_left + models_right
        return StackedLayer(models_stack=models_stack, varin=self.varin)


    # Following are for analysis ----------------------------------------------

    def draw_weight(self, verbose=True, filename='default_draw_layerw.png'):
        """
        Parameters
        -----------
        verbose : bool
        filename : string
    
        Returns
        -----------
        Notes
        -----------
        """
        assert hasattr(self, 'w'), "The layer need to have weight defined."
        if not hasattr(self, '_fig_weight'):
            self.get_w = theano.function([], self.w.T.T)
            self.get_w_cov = theano.function([], T.dot(self.w.T, self.w))

            self._fig_weight = plt.figure(figsize=(7, 11))
            self.plt1 = self._fig_weight.add_subplot(311)
            plt.gray()
            self.p1 = self.plt1.imshow(self.get_w())
            self.plt2 = self._fig_weight.add_subplot(312)
            n, bins, patches = self.plt2.hist(self.get_w().flatten(), 50,
                                              facecolor='blue')
            self.plt3 = self._fig_weight.add_subplot(313)
            self.p3 = self.plt3.imshow(self.get_w_cov())
        else:
            self.p1.set_data(self.get_w())
            self.plt2.cla()
            n, bins, patches = self.plt2.hist(self.get_w().flatten(), 50,
                                              facecolor='blue')
            self.p3.set_data(self.get_w_cov())
        self._fig_weight.canvas.draw()
        
        if verbose:
            plt.pause(0.0001)
        else:
            plt.savefig(filename)

"""
    def dispims_color(M, border=0, bordercolor=[0.0, 0.0, 0.0], *imshow_args, **imshow_keyargs):
    i   """"""" Display an array of rgb images. 

        The input array is assumed to have the shape numimages x numpixelsY x numpixelsX x 3
        """"""
        bordercolor = numpy.array(bordercolor)[None, None, :]
        numimages = len(M)
        M = M.copy()
        for i in range(M.shape[0]):
            M[i] -= M[i].flatten().min()
            M[i] /= M[i].flatten().max()
        height, width, three = M[0].shape
        assert three == 3
        n0 = numpy.int(numpy.ceil(numpy.sqrt(numimages)))
        n1 = numpy.int(numpy.ceil(numpy.sqrt(numimages)))
        im = numpy.array(bordercolor)*numpy.ones(
                             ((height+border)*n1+border,(width+border)*n0+border, 1),dtype='<f8')
        for i in range(n0):
            for j in range(n1):
                if i*n1+j < numimages:
                    im[j*(height+border)+border:(j+1)*(height+border)+border,
                    i*(width+border)+border:(i+1)*(width+border)+border,:] = numpy.concatenate((
                    numpy.concatenate((M[i*n1+j,:,:,:],
                         bordercolor*numpy.ones((height,border,3),dtype=float)), 1),
                         bordercolor*numpy.ones((border,width+border,3),dtype=float)
                    ), 0)
        imshow_keyargs["interpolation"]="nearest"
        pylab.imshow(im, *imshow_args, **imshow_keyargs)
        pylab.show()
"""


class StackedLayer(Layer):
    def __init__(self, models_stack=[], varin=None):
        """
        Ensure the following things:
        1. By calling StackedLayer([...], [...]) we definitely get a stacked la-
           yer.
        2. StackedLayer as a whole can be viewed as a 1-layer model. (i.e. it
           guarantees all features of Layer in an appropriate way.)
        3. By calling
            Layer(...) + Layer(...),
            Layer(...) + StackedLayer(...)
            StackedLayer(...) + Layer(...), or
            StackedLayer(...) + StackedLayer(...)
           we can get a *non-nested* StackedLayer object at its expression
           value.

        Although in the implementation of Layer and StackedLayer class we 
        carefully ensure that nested StackedLayer (i.e. a StackedLayer of
        StackedLayers) does not cause problem, it is not appreaciated because
        it will make it inconvenient for analysing each layer's parameters.
        That's why we flatten the models_stack attribute while overidding "+"
        operator.

        However, it's still possible to create a nested StackedLayer object by
        directly calling the constructor of this class, passing a list with
        elements of StackedLayer objects. ***Avoid to do this unless you have
        a special reasoning on doing it.***
        """
        assert len(models_stack) >= 1, "Warning: A Stacked Layer of empty " + \
                                       "models is trivial."
        for layer in models_stack:
            assert isinstance(layer, Layer), \
                "All models in the models_stack list should be some " + \
                "subclass of Layer"
        super(StackedLayer, self).__init__(
            n_in=models_stack[0].n_in, n_out=models_stack[-1].n_out,
            varin=varin
        )

        previous_layer = None
        for layer_model in models_stack:
            if not previous_layer:  # First layer
                layer_model.varin = self.varin
            else:
                assert previous_layer.n_out == layer_model.n_in, \
                    "Stacked layer should match the input and output " + \
                    "dimension with each other."
                layer_model.varin = previous_layer.output()
            previous_layer = layer_model
            self.params += layer_model.params
        self.models_stack = models_stack

        if hasattr(models_stack[0], 'w'):  # This is for visualizing weights.
            self.w = models_stack[0].w

    def fanin(self):
        """
        The fanin for a StackedLayer is defined as the fanin of its first layer.
        It's automatically in a recursive way, too.
        """
        return self.models_stack[0].fanin()

    def output(self):
        """
        This method is automatically in a recursive way. Think about what will
        happen if we call this function on a StackedLayer object whose last
        layer model is still a StackedLayer object.
        """
        return self.models_stack[-1].output()

    def activ_prime(self):
        # TODO might still exist problem here, if we call this method for 
        # StackedLayer of StackedLayer.
        # Consider change it to raise an error instead?
        return self.models_stack[-1].activ_prime()

    def num_layers(self):
        return len(self.models_stack)

    def print_layer(self):
        print "-" * 35
        print "a stacked model with %d layers:" % self.num_layers()
        print "-" * 35

        previous_layer_string = None
        repeat_count = 0
        for layer_model in self.models_stack:
            layer_string = layer_model._print_str()
            if not previous_layer_string:  # First layer
                num_space = len(layer_string)/2
                print layer_string
            else:
                if layer_string == previous_layer_string:
                    repeat_count += 1
                else:
                    if repeat_count != 0:
                        print " " * (num_space - 5), \
                              "(same x %d)" % (repeat_count + 1)
                        repeat_count = 0
                    print " " * num_space + "|"
                    print layer_string
            previous_layer_string = layer_string
        if repeat_count != 0:
            print " " * (num_space - 5), "same x %d" % (repeat_count + 1)
        print "-" * 35


class SigmoidLayer(Layer):
    def __init__(self, n_in, n_out, varin=None, init_w=None, init_b=None, 
                 npy_rng=None):
        """
        init_w : theano.compile.SharedVariable, optional
            We initialise the weights to be zero here, but it can be initialized
            into a proper random distribution by set_value() in the subclass.
        """
        super(SigmoidLayer, self).__init__(n_in, n_out, varin=varin)
        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)

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
        return T.nnet.sigmoid(self.fanin())

    def activ_prime(self):
        return self.output() * (1. - self.output())


class LinearLayer(Layer):
    def __init__(self, n_in, n_out, varin=None, init_w=None, init_b=None, 
                 npy_rng=None):
        super(LinearLayer, self).__init__(n_in, n_out, varin=varin)
        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)

        if not init_w:
            w = numpy.asarray(npy_rng.uniform(
                low = -4 * numpy.sqrt(6. / (n_in + n_out)),
                high = 4 * numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            init_w = theano.shared(value=w, name='w_linear', borrow=True)
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
                                   name='b_linear', borrow=True)
        else:
            assert init_b.get_value().shape == (n_out,)
        self.b = init_b

        self.params = [self.w, self.b]

    def fanin(self):
        return T.dot(self.varin, self.w) + self.b

    def output(self):
        return self.fanin()

    def activ_prime(self):
        return 1.


class ZerobiasLayer(Layer):
    def __init__(self, n_in, n_out, threshold=1.0, varin=None, init_w=None, 
                 npy_rng=None):
        super(ZerobiasLayer, self).__init__(n_in, n_out, varin=varin)
        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)

        if not init_w:
            w = numpy.asarray(npy_rng.uniform(
                low=-4 * numpy.sqrt(6. / (n_in + n_out)),
                high=4 * numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            init_w = theano.shared(value=w, name='w_zerobias', borrow=True)
        # else:
        #     TODO. The following assetion is complaining about an attribute
        #     error while passing w.T to init_w. Considering using a more
        #     robust way of assertion in the future.
        #     assert init_w.get_value().shape == (n_in, n_out)
        self.w = init_w

        self.params = [self.w]
        self.threshold = theano.shared(
            value=numpy.asarray(threshold, dtype=theano.config.floatX),
            name='zae_threshold',
            borrow=True
        )

    def set_threshold(self, new_threshold):
        self.threshold.set_value(new_threshold)
    
    def fanin(self):
        return T.dot(self.varin, self.w)

    def output(self):
        fanin = self.fanin()
        return T.switch(fanin > self.threshold, fanin, 0.)

    def activ_prime(self):
        return (self.fanin() > self.threshold) * 1. 


class ReluLayer(Layer):
    def __init__(self, n_in, n_out, varin=None, init_w=None, init_b=None, 
                 npy_rng=None):
        super(ReluLayer, self).__init__(n_in, n_out, varin=varin)
        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)

        if not init_w:
            w = numpy.asarray(npy_rng.uniform(
                low = -4 * numpy.sqrt(6. / (n_in + n_out)),
                high = 4 * numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            init_w = theano.shared(value=w, name='w_relu', borrow=True)
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
                                   name='b_relu', borrow=True)
        else:
            assert init_b.get_value().shape == (n_out,)
        self.b = init_b

        self.params = [self.w, self.b]

    def fanin(self):
        return T.dot(self.varin, self.w) + self.b

    def output(self):
        return T.maximum(self.fanin(), 0.)

    def activ_prime(self):
        return (self.fanin() > 0.) * 1.


class TanhLayer(Layer):
    def __init__(self, n_in, n_out, varin=None, init_w=None, init_b=None, 
                 npy_rng=None):
        super(TanhLayer, self).__init__(n_in, n_out, varin=varin)
        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)

        if not init_w:
            w = numpy.asarray(npy_rng.uniform(
                low = -4 * numpy.sqrt(6. / (n_in + n_out)),
                high = 4 * numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            init_w = theano.shared(value=w, name='w_tanh', borrow=True)
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
                                   name='b_tanh', borrow=True)
        else:
            assert init_b.get_value().shape == (n_out,)
        self.b = init_b

        self.params = [self.w, self.b]

    def fanin(self):
        return T.dot(self.varin, self.w) + self.b

    def output(self):
        return T.tanh(self.fanin())

    def activ_prime(self):
        e_m2x = T.exp(-2. * self.fanin())
        return 4. * e_m2x / ((1. + e_m2x) ** 2)


class GatedLinearLayer(Layer):
    def __init__(self):
        raise NotImplementedError("Not implemented yet...")
