"""
All parameters (excluding superparameters) in the model should be in theano var-
iables or theano shared values. In the training part, these variables should be
organized into "theano.function"s. So there should be no theano.function in the 
definition of models here. Except for analysis part of codes.
"""
import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import pool

import matplotlib
matplotlib.use('Agg')
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

        if not hasattr(self, 'params'):
            self.params = []  # to be implemented by subclass

        self.patch_ind = numpy.asarray([])  # needed by plotting.
        self.givens_activation = {}  # hist_activation() is going to need it.

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

    def wTw(self, verbose=True, filename='wTw_layerw.png'):
        assert hasattr(self, 'w'), "The layer need to have weight defined."
        if not hasattr(self, '_wTw'):
            self.get_w_cov = theano.function([], T.dot(self.w.T, self.w))
            self._wTw = plt.figure()
            self.wTw_ax = self._wTw.add_subplot(111)
            plt.gray()
            plt.title(self._print_str() + ': wTw')
            self.wTw_ip = self.wTw_ax.imshow(self.get_w_cov())
        else:
            self.wTw_ip.set_data(self.get_w_cov())
        self._wTw.canvas.draw()
        
        if verbose:
            plt.pause(0.05)
        else:
            plt.savefig(filename)

    def naiveplot_weight(self, verbose=True, 
                         filename='naiveplot_layerw.png'):
        assert hasattr(self, 'w'), "The layer need to have weight defined."
        if not hasattr(self, '_naiveplot_weight'):
            if not hasattr(self, 'get_w'):
                self.get_w = theano.function([], self.w.T.T)
            self._naiveplot_weight = plt.figure()
            self.naive_ax = self._naiveplot_weight.add_subplot(111)
            plt.gray()
            plt.title('weight matrix ' + self._print_str())
            self.naive_ip = self.naive_ax.imshow(self.get_w())
        else:
            self.naive_ip.set_data(self.get_w())
        self._naiveplot_weight.canvas.draw()
        
        if verbose:
            plt.pause(0.05)
        else:
            plt.savefig(filename)

    def hist_weight(self, verbose=True, filename='hist_layerw.png'):
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
        if not hasattr(self, '_hist_weight'):
            if not hasattr(self, 'get_w'):
                self.get_w = theano.function([], self.w.T.T)
            self._hist_weight = plt.figure()
            self.hist_ax = self._hist_weight.add_subplot(111)
        else:
            self.hist_ax.cla()

        n, bins, patches = self.hist_ax.hist(
            self.get_w().flatten(), 50, facecolor='blue'
        )
        self._hist_weight.canvas.draw()
        
        plt.title('weight histogram ' + self._print_str())
        if verbose:
            plt.pause(0.05)
        else:
            plt.savefig(filename)

    def draw_weight(self, patch_shape=None, map_function=None, npatch=None,
                    border=1, bordercolor=(0.0, 0.0, 0.0),
                    verbose=True, filename='draw_layerw.png',
                    *imshow_args, **imshow_keyargs):
        """
        Adapted from Roland Memisevic's code.
        Display an array of images in RGB or grey format. 
        
        Parameters
        -----------
        patch_shape : tuple of integers, has to contain 3 entries.
            First and second entries for the size on X and Y direction, the
            third entry for the number of channels. So the third entry has to
            be 1 for grey image or 3 for RGB image.
        map_function : theano.function
            Maps the weight back through the preprocessing steps to get
            resonable representations of the filters. It should take no inputs
            and output the desired representation of weights.
        npatch : int

        border : int
            Size of the border.
        bordercolor : a tuple of 3 entries.
            Stands for the RGB value of border.
        verbose : bool

        filename : str


        Returns
        -----------


        Notes
        -----------
        Map the weights into a resonable representation by applying it onto
        a theano function 'map_function.' If not given, it displays weights
        by directly reshaping it to the shape specified by patch_shape. The
        weight array should finally be in the shape of:
        npatch x patch_shape[0] x patch_shape[1] x patch_shape[2]

        Here we assume that the map_fun, or the input to the layer if map_fun
        is not provided, are placed in the order of [RRRRGGGGBBBB].
        """
        assert hasattr(self, 'w'), "The layer need to have weight defined."
        if map_function == None:
            if not hasattr(self, 'get_w'):
                self.get_w = theano.function([], self.w.T.T)
            M = self.get_w()
            if len(M.shape) == 2:
                M = M.T
        else:
            assert isinstance(
                map_function,
                theano.compile.function_module.Function
            ), "map_function has to be a theano function with no input."
            M = map_function()

        if len(M.shape) == 2:  # FC weight matrix
            assert M.shape[0] == self.n_out, (
                "Wrong M row numbers. Should be %d " % self.n_out + \
                "but got %d." % M.shape[0])
            max_patchs = self.n_out
        elif len(M.shape) == 4:  # ConvNet filter
            # This will still cause problem for deeper Conv layers.
            assert patch_shape == None, (
                "You don't need to specify a patch_shape for ConvNet layers.")
            patch_shape = (M.shape[2], M.shape[3], M.shape[1])
            max_patchs = M.shape[0]
            M = numpy.swapaxes(M, 1, 3)
            M = numpy.swapaxes(M, 2, 3)

        if npatch == None:
            npatch = max_patchs
        else:
            assert npatch <= max_patchs, ("Too large npatch size. Maximum "
                                          "allowed value %d " % max_patchs + \
                                          "but got %d." % npatch)
            if npatch != len(self.patch_ind):
                if not hasattr(self, 'npy_rng'):
                    self.npy_rng = numpy.random.RandomState(123)
                self.patch_ind = self.npy_rng.permutation(max_patchs)[:npatch]
            M = M[self.patch_ind, :]

        bordercolor = numpy.array(bordercolor)[None, None, :]
        M = M.copy()

        assert patch_shape is not None, ("Need a patch_shape for this case.")
        for i in range(M.shape[0]):
            M[i] -= M[i].flatten().min()
            M[i] /= M[i].flatten().max()
        height, width, channel = patch_shape
        if channel == 1:
            M = numpy.tile(M, 3)
        elif channel != 3:
            raise ValueError(
                "3rd entry of patch_shape has to be either 1 or 3."
            )
        try:
            M = M.reshape((npatch, height, width, 3), order='F')
        except:
            raise ValueError("Wrong patch_shape.")
        
        vpatches = numpy.int(numpy.ceil(numpy.sqrt(npatch)))
        hpatches = numpy.int(numpy.ceil(numpy.sqrt(npatch)))
        vstrike = width + border
        hstrike = height + border
        # initialize image size with border color
        im = numpy.array(bordercolor) * numpy.ones((hstrike * hpatches + border,
                                                    vstrike * vpatches + border,
                                                    1), dtype='<f8')
        for i in range(vpatches):
            for j in range(hpatches):
                if i * hpatches + j < npatch:
                    im[
                        j * hstrike + border:j * hstrike + border + height,
                        i * vstrike + border:i * vstrike + border + width,
                        :
                    ] = M[i * hpatches + j, :, :, :]
        
        if not hasattr(self, '_draw_weight'):
            imshow_keyargs["interpolation"]="nearest"
            self._draw_weight = plt.figure()
            self.draw_ax = self._draw_weight.add_subplot(111)
            self.wimg = self.draw_ax.imshow(im, *imshow_args, **imshow_keyargs)
        else:
            self.wimg.set_data(im)
        self._draw_weight.canvas.draw()

        plt.title('weight plot ' + self._print_str())
        if verbose:
            plt.pause(0.05)
        else:
            plt.savefig(filename)

    def hist_bias(self, verbose=True, filename='hist_layerb.png'):
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
        assert hasattr(self, 'b'), "The layer need to have biases defined."
        if not hasattr(self, '_hist_bias'):
            if not hasattr(self, 'get_b'):
                self.get_b = theano.function([], self.b)
            self._hist_bias = plt.figure()
            self.histb_ax = self._hist_bias.add_subplot(111)
        else:
            self.histb_ax.cla()

        n, bins, patches = self.histb_ax.hist(
            self.get_b().flatten(), 50, facecolor='c'
        )
        self._hist_bias.canvas.draw()
        
        plt.title('bias histogram ' + self._print_str())
        if verbose:
            plt.pause(0.05)
        else:
            plt.savefig(filename)

    def hist_activation(self, givens={}, verbose=True,
                        filename='hist_activation.png'):
        """
        Parameters
        -----------
        givens : dict
            Specify the inputs needed for computing the activation in this
            layer. This will be passed to givens parameter while building
            theano function. Changing this dictionary will result in
            recompiling the function again.
        verbose : bool
        filename : string
    

        Returns
        -----------
        

        Notes
        -----------
        
        """
        if not hasattr(self, '_hist_activation'):
            if not hasattr(self, 'get_activation') or \
            self.givens_activation != givens:
                self.givens_activation = givens
                self.get_activation = theano.function(
                    [], self.output(), givens=givens)
            self._hist_activation = plt.figure()
            self.histact_ax = self._hist_activation.add_subplot(111)
        else:
            self.histact_ax.cla()

        n, bins, patches = self.histact_ax.hist(
            self.get_activation().flatten(), 50, facecolor='red'
        )
        self._hist_activation.canvas.draw()
        
        plt.title('activation histogram ' + self._print_str())
        if verbose:
            plt.pause(0.05)
        else:
            plt.savefig(filename)


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
            Layer(...) + StackedLayer(...),
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

        *Avoid to do the following unless you have a special reasoning on it.*
        However, it's still possible to create a nested StackedLayer object by
        directly calling the constructor of this class, passing a list with
        elements of StackedLayer objects.
        """
        assert len(models_stack) >= 1, ("Warning: A Stacked Layer of empty "
                                        "models is trivial.")
        for layer in models_stack:
            assert isinstance(layer, Layer), (
                "All models in the models_stack list should be some "
                "subclass of Layer")
        
        self.n_in = models_stack[0].n_in
        if not varin:
            varin = T.matrix('varin')
        assert isinstance(varin, T.TensorVariable)
        self.varin = varin
        self.params = []  # to be implemented by subclass

        previous_layer = None
        for layer_model in models_stack:
            if not previous_layer:  # First layer
                layer_model.varin = self.varin
            else:
                right_is_conv = hasattr(layer_model, 'conv')
                left_is_conv = hasattr(previous_layer, 'conv')
                info = ("Dimension mismatch detected when stacking two "
                        "layers.\nformer layer:\n" + \
                        previous_layer._print_str() + \
                        "\nlatter layer:\n")
                # two conv layers or two FC layers
                if (left_is_conv and right_is_conv) or \
                   ((not left_is_conv) and (not right_is_conv)):
                    if layer_model.n_in == None:
                        layer_model.n_in = previous_layer.n_out
                        layer_model._init_complete()
                    assert previous_layer.n_out == layer_model.n_in, \
                        (info + layer_model._print_str() + "\n")
                    layer_model.varin = previous_layer.output()
                elif not right_is_conv:  # CONV-FC
                    if layer_model.n_in == None:
                        layer_model.n_in = numpy.prod(previous_layer.n_out[1:])
                        layer_model._init_complete()
                    assert numpy.prod(previous_layer.n_out[1:]) == \
                        layer_model.n_in, \
                        (info + layer_model._print_str() + "\n")
                    layer_model.varin=previous_layer.output().flatten(2)
                elif not left_is_conv:  # FC-CONV
                    assert numpy.prod(layer_model.n_in) == \
                        previous_layer.n_out[1:], \
                        (info + layer_model._print_str() + "\n")
                    layer_model.varin = \
                        previous_layer.output().reshape(layer_model.n_in)
            previous_layer = layer_model
            self.params += layer_model.params
        self.models_stack = models_stack
        self.n_out = self.models_stack[-1].n_out

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
        print "-" * 50
        print "a stacked model with %d layers:" % self.num_layers()
        print "-" * 50

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
                        print " " * (num_space - 6), \
                              "(+ same x %d)" % (repeat_count)
                        repeat_count = 0
                    print " " * num_space + "|"
                    print layer_string
            previous_layer_string = layer_string
        if repeat_count != 0:
            print " " * (num_space - 5), "(+ same x %d)" % (repeat_count)
        print "-" * 50


class BaseLayer(Layer):
    def __init__(self, n_in, n_out, varin=None, init_w=None, init_b=None, 
                 npy_rng=None):
        """
        An abstract class, providing basic settings for fully connected layers.
        It should be suit for most of the fully connected cases.
        
        n_in : None or int
        You must specify n_in while constructing the object.
        Specify it to be None to let the constructor set it automatically
        according to its previous layer.

        n_out : int

        varin : None or theano symbolic value

        init_w : None or theano.shared

        init_b : None or theano.shared

        npy_rng : None or numpy.random.RandomState

        TODO: WRITEME
        """
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

        self.init_w = init_w
        self.n_in, self.varin = n_in, varin
        if self.n_in != None:
            self._init_complete()

    def _init_complete(self):
        assert self.n_in != None, "Need to have n_in attribute to execute."
        super(BaseLayer, self).__init__(self.n_in, self.n_out,
                                        varin=self.varin)

        if not self.init_w:
            self.init_w = theano.shared(
                value=self._weight_initialization(),
                name=self.__class__.__name__ + '_w', borrow=True)
        # else:
        #     TODO. The following assetion is complaining about an attribute
        #     error while passing w.T to init_w. Considering using a more
        #     robust way of assertion in the future.
        #     assert init_w.get_value().shape == (n_in, n_out)
        self.w = self.init_w

        self.params = [self.w, self.b]

    def _weight_initialization(self):
        return numpy.asarray(
            self.npy_rng.uniform(
                low = -4 * numpy.sqrt(6. / (self.n_in + self.n_out)),
                high = 4 * numpy.sqrt(6. / (self.n_in + self.n_out)),
                size=(self.n_in, self.n_out)),
            dtype=theano.config.floatX
        )

    def fanin(self):
        self.varfanin = T.dot(self.varin, self.w) + self.b
        return self.varfanin

    def output(self, fanin=None):
        raise NotImplementedError("Must be implemented by subclass.")

    def activ_prime(self):
        raise NotImplementedError("Must be implemented by subclass.")

        
class SigmoidLayer(BaseLayer):
    def output(self, fanin=None):
        if fanin == None:   fanin = self.fanin()
        return T.nnet.sigmoid(fanin)

    def activ_prime(self):
        return self.output() * (1. - self.output())


class LinearLayer(BaseLayer):
    def _weight_initialization(self):
        return numpy.asarray(
            self.npy_rng.uniform(
                low = -numpy.sqrt(6. / (self.n_in + self.n_out)),
                high = numpy.sqrt(6. / (self.n_in + self.n_out)),
                size=(self.n_in, self.n_out)),
            dtype=theano.config.floatX
        )

    def output(self, fanin=None):
        if fanin == None:   fanin = self.fanin()
        return fanin

    def activ_prime(self):
        return 1.


class AbsLayer(BaseLayer):
    def output(self, fanin=None):
        if fanin == None:   fanin = self.fanin()
        return T.abs_(fanin)

    def activ_prime(self):
        return (self.fanin() > 0.) * 2. - 1.


class TanhLayer(BaseLayer):
    def _weight_initialization(self):
        return numpy.asarray(
            self.npy_rng.uniform(
                low = -numpy.sqrt(6. / (self.n_in + self.n_out)),
                high = numpy.sqrt(6. / (self.n_in + self.n_out)),
                size=(self.n_in, self.n_out)),
            dtype=theano.config.floatX
        )

    def output(self, fanin=None):
        if fanin == None:   fanin = self.fanin()
        return T.tanh(fanin)

    def activ_prime(self):
        e_m2x = T.exp(-2. * self.fanin())
        return 4. * e_m2x / ((1. + e_m2x) ** 2)


class ReluLayer(BaseLayer):
    def _weight_initialization(self):
        return numpy.asarray(
            self.npy_rng.uniform(
                low = -numpy.sqrt(3. / self.n_in),
                high = numpy.sqrt(3. / self.n_in),
                size=(self.n_in, self.n_out)),
            dtype=theano.config.floatX
        )

    def output(self, fanin=None):
        if fanin == None:   fanin = self.fanin()
        return T.maximum(fanin, 0.)

    def activ_prime(self):
        return (self.fanin() > 0.) * 1.


class BinaryReluLayer(ReluLayer):
    def __init__(self, n_in, n_out, mode='stochastic',
                 varin=None, init_w=None, init_b=None, npy_rng=None):
        super(BinaryReluLayer, self).__init__(n_in=n_in, n_out=n_out, varin=varin,
                         init_w=init_w, init_b=init_b, npy_rng=npy_rng)
        self.srng = theano.sandbox.rng_mrg.MRG_RandomStreams(
            self.npy_rng.randint(999999))
        self.mode = mode

    def binarized_weight(self):
        self.w0 = (numpy.sqrt(3. / self.n_in) / 2).astype(theano.config.floatX)
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

    def fanin(self):
        self.varfanin = T.dot(self.varin, self.binarized_weight()) + self.b
        return self.varfanin

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
        quantized_rep = sign * 2**index_random

        error = T.grad(cost=cost, wrt=self.varfanin)

        self.dEdW = T.dot(quantized_rep.T, error)


class ZerobiasLayer(Layer):
    def __init__(self, n_in, n_out, threshold=1.0, varin=None, init_w=None, 
                 npy_rng=None):
        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)
        self.npy_rng = npy_rng

        self.n_out = n_out
        self.threshold = theano.shared(
            value=numpy.asarray(threshold, dtype=theano.config.floatX),
            name='zae_threshold',
            borrow=True
        )
        
        self.init_w = init_w
        self.n_in, self.varin = n_in, varin
        if self.n_in != None:
            self._init_complete()

    def _init_complete(self):
        assert self.n_in != None, "Need to have n_in attribute to execute."
        super(ZerobiasLayer, self).__init__(self.n_in, self.n_out,
                                            varin=self.varin)
        if not self.init_w:
            w = numpy.asarray(self.npy_rng.uniform(
                low = -numpy.sqrt(6. / (self.n_in + self.n_out)),
                high = numpy.sqrt(6. / (self.n_in + self.n_out)),
                size=(self.n_in, self.n_out)), dtype=theano.config.floatX)
            self.init_w = theano.shared(
                value=w, name=self.__class__.__name__ + '_w', borrow=True)
        self.w = self.init_w

        self.params = [self.w]

    def set_threshold(self, new_threshold):
        self.threshold.set_value(new_threshold)
    
    def fanin(self):
        self.varfanin = T.dot(self.varin, self.w)
        return self.varfanin

    def output(self, fanin=None):
        if fanin == None:   fanin = self.fanin()
        return T.switch(fanin > self.threshold, fanin, 0.)

    def activ_prime(self):
        return (self.fanin() > self.threshold) * 1. 


class PReluLayer(Layer):
    def __init__(self, n_in, n_out, varin=None, init_w=None, init_b=None, 
                 npy_rng=None):
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

        self.lk = theano.shared(
            value=numpy.float32(0.).astype(theano.config.floatX),
            name=self.__class__.__name__ + '_leak_rate'
        )
        
        self.init_w = init_w
        self.n_in, self.varin = n_in, varin
        if self.n_in != None:
            self._init_complete()

    def _init_complete(self):
        assert self.n_in != None, "Need to have n_in attribute to execute."
        super(PReluLayer, self).__init__(self.n_in, self.n_out,
                                         varin=self.varin)
        if not self.init_w:
            w = numpy.asarray(self.npy_rng.uniform(
                low = -numpy.sqrt(3. / self.n_in),
                high = numpy.sqrt(3. / self.n_in),
                size=(self.n_in, self.n_out)), dtype=theano.config.floatX)
            self.init_w = theano.shared(
                value=w, name=self.__class__.__name__ + '_w', borrow=True)
        self.w = self.init_w

        self.params = [self.w, self.b, self.lk]

    def fanin(self):
        self.varfanin = T.dot(self.varin, self.w) + self.b
        return self.varfanin

    def output(self, fanin=None):
        if fanin == None:   fanin = self.fanin()
        return T.switch(fanin > 0., fanin, fanin * self.lk)

    def activ_prime(self):
        return T.switch(self.fanin() > 0., 1., self.lk)


class MaxoutLayer(Layer):
    def __init__(self, n_in, n_piece, n_out, varin=None,
                 init_w=None, init_b=None, npy_rng=None):
        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)
        self.npy_rng = npy_rng

        self.n_out = n_out
        self.n_piece = n_piece
        if not init_b:
            init_b = theano.shared(
                value=numpy.zeros((1, n_piece, n_out), dtype=theano.config.floatX),
                name=self.__class__.__name__ + '_b',
                borrow=True,
                broadcastable=(True, False, False)
            )
        else:
            assert init_b.get_value().shape == (1, n_piece, n_out)
        self.b = init_b

        self.init_w = init_w
        self.n_in, self.varin = n_in, varin
        if self.n_in != None:
            self._init_complete()

    def _init_complete(self):
        assert self.n_in != None, "Need to have n_in attribute to execute."
        super(MaxoutLayer, self).__init__(self.n_in, self.n_out,
                                          varin=self.varin)
        if not self.init_w:
            w = numpy.asarray(
                self.npy_rng.uniform(
                    low = -numpy.sqrt(6. / (self.n_in + self.n_out)),
                    high = numpy.sqrt(6. / (self.n_in + self.n_out)),
                    size=(self.n_in, self.n_piece, self.n_out)),
                dtype=theano.config.floatX
            )
            self.init_w = theano.shared(
                value=w, name=self.__class__.__name__ + '_w', borrow=True)
        self.w = self.init_w

        self.params = [self.w, self.b]

    def fanin(self):
        self.varfanin = T.tensordot(self.varin, self.w, axes=[1, 0]) + self.b
        return self.varfanin

    def output(self, fanin=None):
        if fanin == None:   fanin = self.fanin()
        return T.max(fanin, axis=1)

    def weightdecay(self, weightdecay=1e-3):
        return weightdecay * (self.w**2).sum()
    
    def activ_prime(self):
        raise NotImplementedError("Not implemented yet...")


class GatedLinearLayer(Layer):
    def __init__(self):
        raise NotImplementedError("Not implemented yet...")


class Conv2DLayer(Layer):
    def __init__(self, filter_shape, n_in=None, varin=None,
                 init_w=None, init_b=None, npy_rng=None):
        """
        This is a base class for all 2-D convolutional classes using various of
        activation functions. For more complex ConvNet filters, like network in
        network, don't inherit from this class.

        Parameters
        -----------
        n_in : tuple
        Specifies the dimension of input. The dimension is in bc01 order, i.e.,
        (batch size, # input channels, # input height, # input width)
        The boring thing is that we need to determine batch size, which belongs
        to training and has nothing to do with the model itself, while building
        the model. 

        filter_shape : tuple
        Specifies the filter shape. The dimension is in the order bc01, i.e.
        (# filters, # input channels, # filter height, # filter width)

        varin

        init_w

        init_b

        npy_rng
        
        """
        self.conv = True
        assert len(filter_shape) == 4, ("filter_shape has to be a 4-D tuple "
                                        "ordered in this way: (# filters, "
                                        "# input channels, # filter height, "
                                        "# filter width)")
        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)
        self.npy_rng = npy_rng
        self.filter_shape = filter_shape

        if not init_w:
            numparam_per_filter = numpy.prod(filter_shape[1:])
            w = self._weight_initialization()
            init_w = theano.shared(
                value=w, name=self.__class__.__name__ + '_w', borrow=True)
        # else:
        #     TODO. The following assetion is complaining about an attribute
        #     error while passing w.T to init_w. Considering using a more
        #     robust way of assertion in the future.
        #     assert init_w.get_value().shape == (n_in, n_out)
        self.w = init_w

        if not init_b:
            init_b = theano.shared(
                value=numpy.zeros((filter_shape[0],), dtype=theano.config.floatX),
                name=self.__class__.__name__ + '_b',
                borrow=True
            )
        else:
            assert init_b.get_value().shape == (filter_shape[0],)
        self.b = init_b

        self.n_in, self.varin = n_in, varin
        if self.n_in != None:
            self._init_complete()

    def _init_complete(self):
        assert self.n_in != None, "Need to have n_in attribute to execute."
        assert len(self.n_in) == 4, (
            "n_in is expected to be a 4-D tuple ordered in this way: (batch "
            "size, # input channels, # input height, # input width)")
        # filter_shape[1] has to be the same to n_in[1]
        self.n_out = (self.n_in[0], self.filter_shape[0],
                      self.n_in[2] - self.filter_shape[2] + 1,
                      self.n_in[3] - self.filter_shape[3] + 1)
        if self.n_out[2] <= 0 or self.n_out[3] <= 0:
            raise ValueError(
                "Output dimension of convolution layer reaches 0 :\n" + \
                self._print_str() + "\n"
            )

        super(Conv2DLayer, self).__init__(self.n_in, self.n_out,
                                          varin=self.varin)
        self.varin = self.varin.reshape(self.n_in)       
        self.params = [self.w, self.b]

    def _weight_initialization(self):
        numparam_per_filter = numpy.prod(self.filter_shape[1:])
        w = numpy.asarray(self.npy_rng.uniform(
            low = -numpy.sqrt(3. / numparam_per_filter),
            high = numpy.sqrt(3. / numparam_per_filter),
            size=self.filter_shape), dtype=theano.config.floatX)
        return w

    def fanin(self):
        self.varfanin = conv.conv2d(
            input=self.varin, filters=self.w,
            filter_shape=self.filter_shape, image_shape=self.n_in
        ) + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.varfanin

    def output(self):
        raise NotImplementedError("Must be implemented by subclass.")

    def activ_prime(self):
        raise NotImplementedError("Not implemented yet...")


class LinearConv2DLayer(Conv2DLayer):
    def output(self, fanin=None):
        if fanin == None:   fanin = self.fanin()
        return fanin

    def activ_prime(self):
        raise NotImplementedError("Not implemented yet...")


class ReluConv2DLayer(Conv2DLayer):
    def _weight_initialization(self):
        """
        if there is no 0.1 multiplier, then the variance of each pre-hidden is
        roughly 1.0, which is desired by the principle indicated by batch
        normalization. Adding 0.1 to ensure the initial weights to be
        sufficiently small, which is found to be more efficient in converging
        during the first few epochs. 
        """
        numparam_per_filter = numpy.prod(self.filter_shape[1:])
        w = numpy.asarray(self.npy_rng.uniform(
            low = - 0.1 * numpy.sqrt(3. / numparam_per_filter),
            high = 0.1 * numpy.sqrt(3. / numparam_per_filter),
            size=self.filter_shape), dtype=theano.config.floatX)
        return w

    def output(self, fanin=None):
        if fanin == None:   fanin = self.fanin()
        return T.maximum(fanin, 0.)

    def activ_prime(self):
        raise NotImplementedError("Not implemented yet...")


class Conv2DPoolingReluLayer(Layer):
    def __init__(self, n_in, filter_shape,
                 pool_size, stride=None, ignore_border=False,
                 varin=None, init_w=None, init_b=None, npy_rng=None):
        """
        Order of operation: 
            conv2D -> bias -> max pooling -> ReLU
        If there is batch normalization, it goes between pooling and ReLU.
        """
        self.conv = True

        # convoluiton part
        assert len(filter_shape) == 4, ("filter_shape has to be a 4-D tuple "
                                        "ordered in this way: (# filters, "
                                        "# input channels, # filter height, "
                                        "# filter width)")
        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)
        self.npy_rng = npy_rng
        self.filter_shape = filter_shape
        
        if not init_w:
            numparam_per_filter = numpy.prod(filter_shape[1:])
            w = self._weight_initialization()
            init_w = theano.shared(
                value=w, name=self.__class__.__name__ + '_w', borrow=True)
        self.w = init_w

        if not init_b:
            init_b = theano.shared(
                value=numpy.zeros((filter_shape[0],), dtype=theano.config.floatX),
                name=self.__class__.__name__ + '_b',
                borrow=True
            )
        else:
            assert init_b.get_value().shape == (filter_shape[0],)
        self.b = init_b

        # pooling part:
        assert len(pool_size) == 2, ("pool_size should be a 2-D tuple in the "
                                     "form (# rows, # cols)")
        self.pool_size = pool_size
        if not stride:
            stride = pool_size
        else:
            if len(stride) != 2 or min(stride) <= 0:
                raise ValueError(
                    "stride should be a 2-D tuple in the form (# rows, # cols)"
                    ". Each entry in stride should be strictly larger than 0.")
        self.stride = stride
        self.ignore_border = ignore_border
        
        self.n_in, self.varin = n_in, varin
        if self.n_in != None:
            self._init_complete()

    def _init_complete(self):
        assert self.n_in != None, "Need to have n_in attribute to execute."
        assert len(self.n_in) == 4, (
            "n_in is expected to be a 4-D tuple ordered in this way: (batch "
            "size, # input channels, # input height, # input width)")
        # filter_shape[1] has to be the same to n_in[1]
        conv_out = (self.n_in[0], self.filter_shape[0],
                    self.n_in[2] - self.filter_shape[2] + 1,
                    self.n_in[3] - self.filter_shape[3] + 1)
        if conv_out[2] == 0 or conv_out[3] == 0:
            raise ValueError(
                "Output dimension of convolution layer reaches 0 :\n" + \
                self._print_str() + "\n"
            )

        # use a test method to decide the output dimension, because sometimes
        # the output dimension behaves weirdly. Normally it should be:
        # (max((patch - max(pool-stride, 0)), 1) + stride -1) / stride
        tx = T.matrix().reshape((conv_out[2], conv_out[3]))
        ty = pool.pool_2d(
            tx, ds=self.pool_size,
            ignore_border=self.ignore_border, st=self.stride)

        tf = theano.function([tx], ty)
        out_dim = tf(
            numpy.random.random((conv_out[2], conv_out[3])
            ).astype(theano.config.floatX)
        ).shape

        n_out = (conv_out[0], conv_out[1], out_dim[0], out_dim[1])
        if n_out[2] == 0 or n_out[3] == 0:
            raise ValueError(
                "Output dimension of pooling layer reaches 0 :\n" + \
                self._print_str() + "\n"
            )
        
        super(Conv2DPoolingReluLayer, self).__init__(self.n_in, n_out,
                                                     varin=self.varin)
        self.varin = self.varin.reshape(self.n_in)    
        self.params = [self.w, self.b]

    def _weight_initialization(self):
        """
        if there is no 0.1 multiplier, then the variance of each pre-hidden is
        roughly 1.0, which is desired by the principle indicated by batch
        normalization. Adding 0.1 to ensure the initial weights to be
        sufficiently small, which is found to be more efficient in converging
        during the first few epochs. 
        """
        numparam_per_filter = numpy.prod(self.filter_shape[1:])
        w = numpy.asarray(self.npy_rng.uniform(
            low = - 0.1 * numpy.sqrt(3. / numparam_per_filter),
            high = 0.1 * numpy.sqrt(3. / numparam_per_filter),
            size=self.filter_shape), dtype=theano.config.floatX)
        return w

    def fanin(self):
        varconv = conv.conv2d(
            input=self.varin, filters=self.w,
            filter_shape=self.filter_shape, image_shape=self.n_in
        ) + self.b.dimshuffle('x', 0, 'x', 'x')

        self.varfanin = pool.pool_2d(
            input=varconv, ds=self.pool_size,
            ignore_border=self.ignore_border, st=self.stride)
        return self.varfanin

    def output(self, fanin=None):
        if fanin == None:   fanin = self.fanin()
        return T.maximum(fanin, 0.)

    def activ_prime(self):
        raise NotImplementedError("Not implemented yet...")


class BinaryConv2DPoolingReluLayer(Conv2DPoolingReluLayer):
    def __init__(self, n_in, filter_shape, pool_size, stride=None,
                 ignore_border=False, varin=None, mode='stochastic',
                 init_w=None, init_b=None, npy_rng=None):
        super(BinaryConv2DPoolingReluLayer, self).__init__(
            filter_shape=filter_shape, pool_size=pool_size, stride=stride,
            n_in=n_in, varin=varin,
            init_w=init_w, init_b=init_b, npy_rng=npy_rng
        )
        self.srng = theano.sandbox.rng_mrg.MRG_RandomStreams(
            self.npy_rng.randint(999999))
        self.mode = mode
    
    def binarized_weight(self):
        self.w0 = (
            numpy.sqrt(3. / numpy.prod(self.filter_shape[1:])) / 2
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

    def fanin(self):
        self.varconv = T.nnet.conv.conv2d(
            input=self.varin, filters=self.binarized_weight(),
            filter_shape=self.filter_shape, image_shape=self.n_in
        ) + self.b.dimshuffle('x', 0, 'x', 'x')

        self.varfanin = pool.pool_2d(
            input=self.varconv, ds=self.pool_size,
            ignore_border=self.ignore_border, st=self.stride)
        return self.varfanin

    def quantized_bprop(self, cost):
        """
        bprop for convolution layer equals:
        
        (
            self.x.dimshuffle(1, 0, 2, 3)       (*) 
            T.grad(cost, wrt=#convoutput).dimshuffle(1, 0, 2, 3)[:, :, ::-1, ::-1]
        ).dimshuffle(1, 0, 2, 3)[:, :, ::-1, ::-1]

        '(*)'stands for convolution.
        Here we quantize (rep of previous layer) and leave the rest as it is.
        """
        # the lower 2**(integer power)
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
        quantized_rep = sign * 2**index_random
        
        error = T.grad(cost=cost, wrt=self.varconv)

        self.dEdW = T.nnet.conv.conv2d(
            input=quantized_rep.dimshuffle(1, 0, 2, 3),
            filters=error.dimshuffle(1, 0, 2, 3)[:, :, ::-1, ::-1]
        ).dimshuffle(1, 0, 2, 3)[:, :, ::-1, ::-1]


class BinaryReluConv2DLayer(ReluConv2DLayer):
    def __init__(self, n_in, filter_shape, mode='stochastic',
                 varin=None, init_w=None, init_b=None, npy_rng=None):
        super(BinaryReluConv2DLayer, self).__init__(
            filter_shape=filter_shape, n_in=n_in, varin=varin,
            init_w=init_w, init_b=init_b, npy_rng=npy_rng
        )
        self.srng = theano.sandbox.rng_mrg.MRG_RandomStreams(
            self.npy_rng.randint(999999))
        self.mode = mode

    def binarized_weight(self):
        self.w0 = (
            numpy.sqrt(3. / numpy.prod(self.filter_shape[1:])) / 2
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

    def fanin(self):
        self.varfanin = T.nnet.conv.conv2d(
            input=self.varin, filters=self.binarized_weight(),
            filter_shape=self.filter_shape, image_shape=self.n_in
        ) + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.varfanin

    def quantized_bprop(self, cost):
        """
        bprop for convolution layer equals:
        
        (
            self.x.dimshuffle(1, 0, 2, 3)       (*) 
            T.grad(cost, wrt=#convoutput).dimshuffle(1, 0, 2, 3)[:, :, ::-1, ::-1]
        ).dimshuffle(1, 0, 2, 3)[:, :, ::-1, ::-1]

        '(*)'stands for convolution.
        Here we quantize (rep of previous layer) and leave the rest as it is.
        """
        # the lower 2**(integer power)
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
        quantized_rep = sign * 2**index_random
        
        error = T.grad(cost=cost, wrt=self.varfanin)

        self.dEdW = T.nnet.conv.conv2d(
            input=quantized_rep.dimshuffle(1, 0, 2, 3),
            filters=error.dimshuffle(1, 0, 2, 3)[:, :, ::-1, ::-1]
        ).dimshuffle(1, 0, 2, 3)[:, :, ::-1, ::-1]


class PReluConv2DLayer(Conv2DLayer):
    def __init__(self, filter_shape, n_in=None, varin=None,
                 init_w=None, init_b=None, npy_rng=None):
        self.lk = theano.shared(
            value=numpy.float32(0.).astype(theano.config.floatX),
            name=self.__class__.__name__ + '_leak_rate'
        )
        super(PReluConv2DLayer, self).__init__(
            filter_shape, n_in=n_in, varin=varin, init_w=init_w,
            init_b=init_b, npy_rng=npy_rng
        )

    def _init_complete(self):
        super(PReluConv2DLayer, self)._init_complete()
        self.params.append(self.lk)

    def _weight_initialization(self):
        numparam_per_filter = numpy.prod(self.filter_shape[1:])
        w = numpy.asarray(self.npy_rng.uniform(
            low = - 0.1 * numpy.sqrt(3. / numparam_per_filter),
            high = 0.1 * numpy.sqrt(3. / numparam_per_filter),
            size=self.filter_shape), dtype=theano.config.floatX)
        return w

    def output(self, fanin=None):
        if fanin == None:   fanin = self.fanin()
        return T.switch(fanin > 0., fanin, fanin * self.lk)

    def activ_prime(self):
        raise NotImplementedError("Not implemented yet...")


class PoolingLayer(Layer):
    def __init__(self, pool_size, n_in=None, stride=None, ignore_border=False,
                 varin=None):
        """
        Parameters
        -----------
        n_in : tuple
        Specifies the dimension of input. The dimension is in bc01 order, i.e.,
        (batch size, # input channels, # input height, # input width)
        The boring thing is that we need to determine batch size, which belongs
        to training and has nothing to do with the model itself, while building
        the model. 

        pool_size : tuple
        A tuple with 2 entries.

        stride : tuple
        A tuple with 2 entries, each indicating the pooling stride on each
        dimension.

        ignore_border : bool

        varin :

        """
        self.conv = True
        assert len(pool_size) == 2, ("pool_size should be a 2-D tuple in the "
                                     "form (# rows, # cols)")
        self.pool_size = pool_size
        if not stride:
            stride = pool_size
        else:
            if len(stride) != 2 or min(stride) <= 0:
                raise ValueError(
                    "stride should be a 2-D tuple in the form (# rows, # cols)"
                    ". Each entry in stride should be strictly larger than 0.")
        self.stride = stride
        self.ignore_border = ignore_border
        
        self.n_in, self.varin = n_in, varin
        if self.n_in != None:
            self._init_complete()

    def _init_complete(self):
        assert self.n_in != None, "Need to have n_in attribute to execute."
        assert len(self.n_in) == 4, ("n_in is expected to be a 4-D tuple "
                                     "ordered in this way: (batch size, # "
                                     "input channels, # input height, # "
                                     "input width)")
        
        # use a test method to decide the output dimension, because sometimes
        # the output dimension behaves weirdly. Normally it should be:
        # (max((patch - max(pool-stride, 0)), 1) + stride -1) / stride
        tx = T.matrix().reshape((self.n_in[2], self.n_in[3]))
        ty = pool.pool_2d(
            tx, ds=self.pool_size,
            ignore_border=self.ignore_border, st=self.stride)

        tf = theano.function([tx], ty)
        out_dim = tf(
            numpy.random.random((self.n_in[2], self.n_in[3])
            ).astype(theano.config.floatX)
        ).shape

        n_out = (self.n_in[0], self.n_in[1], out_dim[0], out_dim[1])
        if n_out[2] == 0 or n_out[3] == 0:
            raise ValueError(
                "Output dimension of pooling layer reaches 0 :\n" + \
                self._print_str() + "\n"
            )
        
        super(PoolingLayer, self).__init__(n_in=self.n_in, n_out=n_out,
                                           varin=self.varin)
        self.varin = self.varin.reshape(self.n_in)

    def fanin(self):
        raise NotImplementedError("Must be implemented by subclass.")
    
    def output(self, fanin=None):
        if fanin == None:   fanin = self.fanin()
        return fanin

    def activ_prime(self):
        raise NotImplementedError("Not implemented yet...")


class MaxPoolingLayer(PoolingLayer):
    def fanin(self):
        return pool.pool_2d(
            input=self.varin, ds=self.pool_size,
            ignore_border=self.ignore_border, st=self.stride
        )
