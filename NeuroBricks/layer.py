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
from theano.tensor.signal import downsample

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
        super(StackedLayer, self).__init__(
            n_in=models_stack[0].n_in, n_out=models_stack[-1].n_out,
            varin=varin
        )

        previous_layer = None
        for layer_model in models_stack:
            if not previous_layer:  # First layer
                layer_model.varin = self.varin
            else:
                left_is_tuple = isinstance(previous_layer.n_out, tuple)
                right_is_tuple = isinstance(layer_model.n_in, tuple)
                info = ("Dimension mismatch detected when stacking two "
                        "layers.\nformer layer:\n" + \
                        previous_layer._print_str() + \
                        "\nlatter layer:\n" + layer_model._print_str() + "\n")
                # two conv layers or two FC layers
                if (left_is_tuple and right_is_tuple) or \
                   ((not left_is_tuple) and (not right_is_tuple)):
                    assert previous_layer.n_out == layer_model.n_in, info
                    layer_model.varin = previous_layer.output()
                elif left_is_tuple:  # CONV-FC
                    assert numpy.prod(previous_layer.n_out[1:]) == \
                        layer_model.n_in, info
                    layer_model.varin=previous_layer.output().flatten(2)
                elif right_is_tuple:  # FC-CONV
                    assert numpy.prod(layer_model.n_in) == \
                        previous_layer.n_out[1:], info
                    layer_model.varin = \
                        previous_layer.output().reshape(previous_layer.n_out)
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
        print "-" * 40
        print "a stacked model with %d layers:" % self.num_layers()
        print "-" * 40

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
                low = -numpy.sqrt(6. / (n_in + n_out)),
                high = numpy.sqrt(6. / (n_in + n_out)),
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
                low = -numpy.sqrt(6. / (n_in + n_out)),
                high = numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            init_w = theano.shared(value=w, name='w_zerobias', borrow=True)
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
                low = -numpy.sqrt(3. / n_in),
                high = numpy.sqrt(3. / n_in),
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


class PReluLayer(Layer):
    def __init__(self, n_in, n_out, varin=None, init_w=None, init_b=None, 
                 npy_rng=None):
        super(PReluLayer, self).__init__(n_in, n_out, varin=varin)
        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)

        if not init_w:
            w = numpy.asarray(npy_rng.uniform(
                low = -numpy.sqrt(3. / n_in),
                high = numpy.sqrt(3. / n_in),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            init_w = theano.shared(value=w, name='w_relu', borrow=True)
        self.w = init_w

        if not init_b:
            init_b = theano.shared(value=numpy.zeros(
                                       n_out,
                                       dtype=theano.config.floatX),
                                   name='b_relu', borrow=True)
        else:
            assert init_b.get_value().shape == (n_out,)
        self.b = init_b

        self.lk = theano.shared(
            value=numpy.float32(0.).astype(theano.config.floatX),
            name='leak_rate'
        )
        
        self.params = [self.w, self.b, self.lk]

    def fanin(self):
        return T.dot(self.varin, self.w) + self.b

    def output(self):
        fanin = self.fanin()
        return T.switch(fanin > 0., fanin, fanin * self.lk)

    def activ_prime(self):
        return T.switch(self.fanin() > 0., 1., self.lk)


class AbsLayer(Layer):
    def __init__(self, n_in, n_out, varin=None, init_w=None, init_b=None, 
                 npy_rng=None):
        super(AbsLayer, self).__init__(n_in, n_out, varin=varin)
        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)

        if not init_w:
            w = numpy.asarray(npy_rng.uniform(
                low = -4 * numpy.sqrt(6. / (n_in + n_out)),
                high = 4 * numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            init_w = theano.shared(value=w, name='w_abs', borrow=True)
        # else:
        #     TODO. The following assetion is complaining about an attribute
        #     error while passing w.T to init_w. Considering using a more
        #     robust way of assertion in the future.
        #     assert init_w.get_value().shape == (n_in, n_out)
        self.w = init_w

        if not init_b:
            init_b = theano.shared(
                value=numpy.zeros(n_out, dtype=theano.config.floatX),
                name='b_abs',
                borrow=True
            )
        else:
            assert init_b.get_value().shape == (n_out,)
        self.b = init_b

        self.params = [self.w, self.b]

    def fanin(self):
        return T.dot(self.varin, self.w) + self.b

    def output(self):
        return T.abs_(self.fanin())

    def activ_prime(self):
        return (self.fanin() > 0.) * 2. - 1.


class TanhLayer(Layer):
    def __init__(self, n_in, n_out, varin=None, init_w=None, init_b=None, 
                 npy_rng=None):
        super(TanhLayer, self).__init__(n_in, n_out, varin=varin)
        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)

        if not init_w:
            w = numpy.asarray(npy_rng.uniform(
                low = -numpy.sqrt(6. / (n_in + n_out)),
                high = numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            init_w = theano.shared(value=w, name='w_tanh', borrow=True)
        # else:
        #     TODO. The following assetion is complaining about an attribute
        #     error while passing w.T to init_w. Considering using a more
        #     robust way of assertion in the future.
        #     assert init_w.get_value().shape == (n_in, n_out)
        self.w = init_w

        if not init_b:
            init_b = theano.shared(
                value=numpy.zeros(n_out, dtype=theano.config.floatX),
                name='b_tanh',
                borrow=True
            )
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


class Conv2DLayer(Layer):
    def __init__(self, n_in, filter_shape, n_out=None, varin=None,
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

        n_out : tuple
        Specifies the dimension of output. Should be consistant to the
        convolution of filter_shape and n_in.

        filter_shape : tuple
        Specifies the filter shape. The dimension is in the order bc01, i.e.
        (# filters, # input channels, # filter height, # filter width)

        varin

        init_w

        init_b

        npy_rng
        
        """
        assert len(filter_shape) == 4, ("filter_shape has to be a 4-D tuple "
                                        "ordered in this way: (# filters, "
                                        "# input channels, # filter height, "
                                        "# filter width)")
        assert len(n_in) == 4, ("n_in is expected to be a 4-D tuple ordered "
                                "in this way: (batch size, # input channels, "
                                "# input height, # input width)")
        # filter_shape[1] has to be the same to n_in[1]
        n_out_calcu = (n_in[0], filter_shape[0],
                       n_in[2] - filter_shape[2] + 1,
                       n_in[3] - filter_shape[3] + 1)
        if not n_out:
            n_out = n_out_calcu
        assert n_out == n_out_calcu, "Given n_out doens't match actual output."
        super(Conv2DLayer, self).__init__(n_in, n_out, varin=varin)
        self.varin = self.varin.reshape(self.n_in)
        self.filter_shape = filter_shape
        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)
        self.npy_rng = npy_rng

        if not init_w:
            numparam_per_filter = numpy.prod(filter_shape[1:])
            w = self._weight_initialization()
            init_w = theano.shared(value=w, name='w_conv2d', borrow=True)
        # else:
        #     TODO. The following assetion is complaining about an attribute
        #     error while passing w.T to init_w. Considering using a more
        #     robust way of assertion in the future.
        #     assert init_w.get_value().shape == (n_in, n_out)
        self.w = init_w

        if not init_b:
            init_b = theano.shared(
                value=numpy.zeros((filter_shape[0],), dtype=theano.config.floatX),
                name='b_conv2d',
                borrow=True
            )
        else:
            assert init_b.get_value().shape == (filter_shape[0],)
        self.b = init_b

        self.params = [self.w, self.b]
        

    def _weight_initialization(self):
        numparam_per_filter = numpy.prod(self.filter_shape[1:])
        w = numpy.asarray(self.npy_rng.uniform(
            low = -numpy.sqrt(3. / numparam_per_filter),
            high = numpy.sqrt(3. / numparam_per_filter),
            size=self.filter_shape), dtype=theano.config.floatX)
        return w

    def fanin(self):
        return conv.conv2d(
            input=self.varin, filters=self.w,
            filter_shape=self.filter_shape, image_shape=self.n_in
        ) + self.b.dimshuffle('x', 0, 'x', 'x')

    def output(self):
        raise NotImplementedError("Must be implemented by subclass.")

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

    def output(self):
        return T.maximum(self.fanin(), 0.)

    def activ_prime(self):
        raise NotImplementedError("Not implemented yet...")


class PReluConv2DLayer(Conv2DLayer):
    def __init__(self, n_in, filter_shape, n_out=None, varin=None,
                 init_w=None, init_b=None, npy_rng=None):
        super(PReluConv2DLayer, self).__init__(
            n_in, filter_shape, n_out=None, varin=None, init_w=None,
            init_b=None, npy_rng=None
        )
        self.lk = theano.shared(
            value=numpy.float32(0.).astype(theano.config.floatX),
            name='leak_rate'
        )
        self.params.append(self.lk)

    def _weight_initialization(self):
        numparam_per_filter = numpy.prod(self.filter_shape[1:])
        w = numpy.asarray(self.npy_rng.uniform(
            low = - 0.1 * numpy.sqrt(3. / numparam_per_filter),
            high = 0.1 * numpy.sqrt(3. / numparam_per_filter),
            size=self.filter_shape), dtype=theano.config.floatX)
        return w

    def output(self):
        fanin = self.fanin()
        return T.switch(fanin > 0., fanin, fanin * self.lk)

    def activ_prime(self):
        raise NotImplementedError("Not implemented yet...")


class PoolingLayer(Layer):
    def __init__(self, n_in, pool_size, ignore_border=False, n_out=None,
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

        pool_size :

        ignore_border :
        
        n_out : tuple
        Specifies the dimension of output. Should be consistant to the
        convolution of filter_shape and n_in.

        varin

        """
        assert len(n_in) == 4, ("n_in is expected to be a 4-D tuple ordered in "
                                "this way: (batch size, # input channels, "
                                "# input height, # input width)")
        assert len(pool_size) == 2, ("pool_size should be a 2-D tuple in the "
                                     "form (# rows, # cols)")
        n_out_calcu = (n_in[0], n_in[1], n_in[2] / pool_size[0],
                       n_in[3] / pool_size[1])
        if not n_out:
            n_out = n_out_calcu
        assert n_out == n_out_calcu, "Given n_out doens't match actual output."
        super(PoolingLayer, self).__init__(n_in, n_out, varin=varin)
        self.varin = self.varin.reshape(self.n_in)
        self.pool_size = pool_size
        self.ignore_border = ignore_border

    def output(self):
        raise NotImplementedError("Must be implemented by subclass.")

    def activ_prime(self):
        raise NotImplementedError("Not implemented yet...")


class MaxPoolingLayer(PoolingLayer):
    def output(self):
        return downsample.max_pool_2d(
            input=self.varin, ds=self.pool_size,
            ignore_border=self.ignore_border
        )

    def activ_prime(self):
        raise NotImplementedError("Not implemented yet...")
