import copy
import time
import numpy
import numpy.random
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

SharedCPU = theano.tensor.sharedvar.TensorSharedVariable
try:
    SharedGPU = theano.sandbox.cuda.var.CudaNdarraySharedVariable
except:
    SharedGPU = SharedCPU

from layer import Layer, StackedLayer
from model import AutoEncoder

import pdb


class GraddescentMinibatch(object):
    def __init__(self, varin, data, cost, params, 
                 truth=None, truth_data=None, supervised=False,
                 batchsize=100, learningrate=0.1, momentum=0.9, 
                 rng=None, verbose=True):
        """
        Using stochastic gradient descent with momentum on data in a minibatch
        update manner.
        """
        
        # TODO: check dependencies between varin, cost, and param.
        
        assert isinstance(varin, T.TensorVariable)
        if (not isinstance(data, SharedCPU)) and \
           (not isinstance(data, SharedGPU)):
            raise TypeError("\'data\' needs to be a theano shared variable.")
        assert isinstance(cost, T.TensorVariable)
        assert isinstance(params, list)
        self.varin         = varin
        self.data          = data
        self.cost          = cost
        self.params        = params
        
        if supervised:
            if (not isinstance(truth_data, SharedCPU)) and \
               (not isinstance(truth_data, SharedGPU)):
                raise TypeError("\'truth_data\' needs to be a theano " + \
                                "shared variable.")
            assert isinstance(truth, T.TensorVariable)
            self.truth_data = truth_data
            self.truth = truth
        
        self.verbose       = verbose
        self.batchsize     = batchsize
        self.numbatches    = self.data.get_value().shape[0] / batchsize
        self.momentum      = momentum 
        self.supervised    = supervised
        
        if rng is None:
            rng = numpy.random.RandomState(1)
        assert isinstance(rng, numpy.random.RandomState), \
            "rng has to be a random number generater."
        self.rng = rng

        self.epochcount = 0
        self.stepcount = 0
        self.stepcost = 0.
        self.steptimer = 0.
        
        self.index = T.lscalar('batch_index_in_sgd') 
        self.incs = dict([(
            p, 
            theano.shared(value=numpy.zeros(p.get_value().shape, 
                                            dtype=theano.config.floatX),
                          name='inc_' + p.name,
                          broadcastable=p.broadcastable)
        ) for p in self.params])

        self.grad = T.grad(self.cost, self.params)

        self.set_learningrate(learningrate)
        
        params_vector = T.concatenate([p.flatten() for p in self.params])
        self.get_params_value = theano.function([], params_vector)
        self.ref_vector = theano.shared(value=self.get_params_value(),
                                        name='step_ref_params',
                                        borrow=True)

        inc_vector = params_vector - self.ref_vector
        norm_ref_vector = T.sqrt(T.sum(self.ref_vector ** 2))
        norm_inc_vector = T.sqrt(T.sum(inc_vector ** 2))
        angle_rad = T.arccos(T.dot(self.ref_vector, inc_vector) /\
                    (norm_ref_vector * norm_inc_vector))

        self.get_step_info = theano.function([], (norm_inc_vector, angle_rad))

    def set_learningrate(self, learningrate):
        self.learningrate  = learningrate
        self.inc_updates = []  # updates the parameter increasements (i.e. 
                               # value in the self.incs dictionary.). Due to 
                               # momentum, the increasement itself is
                               # changing between epochs. Its increasing by:
                               # from (key) inc_params 
                               # to (value) momentum * inc_params - lr * grad
                               
        self.updates = []  # updates the parameters of model during each epoch.
                           # from (key) params
                           # to (value) params + inc_params
                           
        for _param, _grad in zip(self.params, self.grad):
            self.inc_updates.append(
                (self.incs[_param],
                 self.momentum * self.incs[_param] - self.learningrate * _grad
                )
            )
            self.updates.append((_param, _param + self.incs[_param]))

        if not self.supervised:
            self._updateincs = theano.function(
                inputs = [self.index], 
                outputs = self.cost, 
                updates = self.inc_updates,
                givens = {
                    self.varin : self.data[self.index * self.batchsize: \
                                           (self.index+1)*self.batchsize]
                }
            )
        else:
            self._updateincs = theano.function(
                inputs = [self.index],
                outputs = self.cost,
                updates = self.inc_updates,
                givens = {
                    self.varin : self.data[self.index * self.batchsize: \
                                           (self.index+1)*self.batchsize],
                    self.truth : self.truth_data[self.index * self.batchsize: \
                                                 (self.index+1)*self.batchsize]
                }
            )

        #self.n = T.scalar('n')
        #self.noop = 0.0 * self.n
        #self._trainmodel = theano.function([self.n], self.noop, 
        #                                   updates = self.updates)
        self._trainmodel = theano.function(inputs=[], updates = self.updates)
            

    def epoch(self):
        start = time.time()
        stepcount = 0.0
        cost = 0.
        self.ref_vector.set_value(self.get_params_value())
        for batch_index in self.rng.permutation(self.numbatches - 1):
            stepcount += 1.0
            # This is Roland's way of computing cost, still mean over all
            # batches. It saves space and don't harm computing time... 
            # But a little bit unfamilliar to understand at first glance.
            cost = (1.0 - 1.0/stepcount) * cost + \
                   (1.0/stepcount) * self._updateincs(batch_index)
            
            self._trainmodel()

        norm, angle_rad = self.get_step_info()
        self.epochcount += 1
        stop = time.time()
        if self.verbose:
            print 'epoch %d: %.2fs, lr %.3g cost %.6g, ' % (
                self.epochcount, (stop - start), self.learningrate, cost) + \
                  'update norm %.3g angle(RAD) %.3f' % (norm, angle_rad)

        return cost

    def step(self, verbose_stride=1):
        """
        Randomly pick a minibatch from dataset, and perform one step of update.
        
        If you switch this method between self.epoch() during training, the
        update norm, angle may not be immediately correct after the epoch/step
        at which you switch.
        """
        start = time.time()
        self.ref_vector.set_value(self.get_params_value())
        batch_index = self.rng.randint(0, self.numbatches - 1)
        
        cost = self._updateincs(batch_index)
        self._trainmodel()
        
        self.stepcost += cost
        self.stepcount += 1

        norm, angle_rad = self.get_step_info()
        stop = time.time()
        self.steptimer += (stop - start)
        if (self.stepcount % verbose_stride == 0) and self.verbose:
            print 'minibatch %d: %.2fs, lr %.3g cost %.6g, ' % (
                self.stepcount, self.steptimer, self.learningrate,
                self.stepcost / verbose_stride) + \
                  'update norm %.3g angle(RAD) %.3f' % (norm, angle_rad)
            self.steptimer = 0.
            self.stepcost = 0.
        return cost

    def step_fast(self, verbose_stride=1):
        """
        A faster implementation of step(). Removes evaluation of angles,
        norms, etc.
        
        MUCH FASTER!! ~ 20 times!!
        """
        start = time.time()
        batch_index = self.rng.randint(0, self.numbatches - 1)
        
        cost = self._updateincs(batch_index)
        self._trainmodel()
        
        self.stepcost += cost
        self.stepcount += 1

        stop = time.time()
        self.steptimer += (stop - start)
        if (self.stepcount % verbose_stride == 0) and self.verbose:
            print 'minibatch %d: %.2fs, lr %.3g cost %.6g ' % (
                self.stepcount, self.steptimer, self.learningrate,
                self.stepcost / verbose_stride)
            self.steptimer = 0.
            self.stepcost = 0.
        return cost

    def draw_gradient(self,):
        raise NotImplementedError("Not implemented yet...")


class ConjugateGradient(object):
    def __init__(self, varin, data, cost, params, 
                 truth=None, truth_data=None, supervised=False,
                 batchsize=100, learningrate=0.1, momentum=0.9, 
                 rng=None, verbose=True):
        """
        Using stochastic gradient descent with momentum on data in a minibatch
        update manner.
        """
        print "ERROR: Not usable now."


class FeedbackAlignment(object):
    def __init__(self, model, data, truth_data, 
                 batchsize=100, learningrate=0.1, rng=None, verbose=True):
        """
        It works for both linear and nonlinear layers, as long as they have 
        the activ_prime() method.

        Cost is defined intrinsicaly as the MSE between target y vector and 
        real y vector at the top layer.

        Parameters:
        ------------
        model : StackedLayer
        data : theano.compile.SharedVariable
        truth_data : theano.compile.SharedVariable

        Notes:
        ------------
        """
        if (not isinstance(data, SharedCPU)) and \
           (not isinstance(data, SharedGPU)):
            raise TypeError("\'data\' needs to be a theano shared variable.")
        if (not isinstance(truth_data, SharedCPU)) and \
           (not isinstance(truth_data, SharedGPU)):
            raise TypeError("\'truth_data\' needs to be a theano shared variable.")
        self.varin         = model.models_stack[0].varin
        self.truth         = T.lmatrix('trurh_fba')
        self.data          = data
        self.truth_data    = truth_data

        self.model         = model
        self.output        = model.models_stack[-1].output()
        self.verbose       = verbose
        self.batchsize     = batchsize
        self.numbatches    = self.data.get_value().shape[0] / batchsize
 
        if rng is None:
            rng = numpy.random.RandomState(1)
        assert isinstance(rng, numpy.random.RandomState), \
            "rng has to be a random number generater."
        self.rng = rng

        self.error = (self.truth - self.output) * \
                     self.model.models_stack[-1].activ_prime()

        # set fixed random matrix
        self.fixed_B = [None, ]
        for imod in self.model.models_stack[1:]:
            i_layer_B = []
            for ipar in imod.params:
                rnd = numpy.asarray(
                    self.rng.uniform(
                        low = -4 * numpy.sqrt(6. / (imod.n_in + imod.n_out)),
                        high = 4 * numpy.sqrt(6. / (imod.n_in + imod.n_out)),
                        size = ipar.get_value().shape
                    ), 
                    dtype=ipar.dtype
                )
 
                i_layer_B.append(
                    theano.shared(value = rnd, name=ipar.name + '_fixed',
                                  borrow=True)
                )
            self.fixed_B.append(i_layer_B)

        self.epochcount = 0
        self.index = T.lscalar('batch_index_in_fba') 
        self._get_cost = theano.function(
            inputs = [self.index],
            outputs = T.sum(self.error ** 2),
            givens = {
                 self.varin : self.data[self.index * self.batchsize: \
                                        (self.index+1)*self.batchsize],
                 self.truth : self.truth_data[self.index * self.batchsize: \
                                              (self.index+1)*self.batchsize]
            }
        )

        self.set_learningrate(learningrate)


    def set_learningrate(self, learningrate):
        self.learningrate = learningrate

        layer_error = self.error
        self.layer_learning_funcs = []
        for i in range(len(self.model.models_stack) - 1, -1, -1):
            iupdates = []
            iupdates.append((
                 self.model.models_stack[i].w,
                 self.model.models_stack[i].w + self.learningrate * \
                     T.dot(self.model.models_stack[i].varin.T, layer_error)
            ))  # w
            iupdates.append((
                 self.model.models_stack[i].b,
                 self.model.models_stack[i].b + self.learningrate * \
                     T.mean(layer_error, axis=0)
            ))  # b
            if i > 0:  # exclude the first layer.
                layer_error = T.dot(layer_error, self.fixed_B[i][0].T) * \
                    self.model.models_stack[i-1].activ_prime()
            
            self.layer_learning_funcs.append(
                theano.function(
                    inputs = [self.index],
                    outputs = self.model.models_stack[i].output(),
                    updates = iupdates,
                    givens = {
                        self.varin : self.data[
                            self.index * self.batchsize: \
                            (self.index+1)*self.batchsize
                        ],
                        self.truth : self.truth_data[
                            self.index * self.batchsize: \
                            (self.index+1)*self.batchsize
                        ]
                    }
                )
            )  

    def epoch(self):
        stepcount = 0.
        cost = 0.
        for batch_index in self.rng.permutation(self.numbatches - 1):
            stepcount += 1.
            cost = (1.0 - 1.0/stepcount) * cost + \
                   (1.0/stepcount) * self._get_cost(batch_index)
            for layer_learn in self.layer_learning_funcs:
                layer_learn(batch_index)

        self.epochcount += 1
        if self.verbose:
            print 'epoch: %d, lr: %f, cost: %f' % (
                self.epochcount, self.learningrate, cost
            )
        return cost


class QuantizedBackProp(object):
    def __init__(self, model, varin, data, cost, params, 
                 truth=None, truth_data=None, supervised=False,
                 batchsize=100, learningrate=0.1, momentum=0.9, 
                 rng=None, verbose=True):
        """
        Using stochastic gradient descent with momentum on data in a minibatch
        update manner.
        It requires ALL layers 
        """
        assert isinstance(model, StackedLayer)
        assert isinstance(varin, T.TensorVariable)
        if (not isinstance(data, SharedCPU)) and \
           (not isinstance(data, SharedGPU)):
            raise TypeError("\'data\' needs to be a theano shared variable.")
        assert isinstance(cost, T.TensorVariable)
        assert isinstance(params, list)
        self.model         = model
        self.varin         = varin
        self.data          = data
        self.cost          = cost
        self.params        = params
        
        if supervised:
            if (not isinstance(truth_data, SharedCPU)) and \
               (not isinstance(truth_data, SharedGPU)):
                raise TypeError("\'truth_data\' needs to be a theano " + \
                                "shared variable.")
            assert isinstance(truth, T.TensorVariable)
            self.truth_data = truth_data
            self.truth = truth
        
        self.verbose       = verbose
        self.batchsize     = batchsize
        self.numbatches    = self.data.get_value().shape[0] / batchsize
        self.momentum      = momentum 
        self.supervised    = supervised
        
        if rng is None:
            rng = numpy.random.RandomState(1)
        assert isinstance(rng, numpy.random.RandomState), \
            "rng has to be a random number generater."
        self.rng = rng

        self.epochcount = 0
        self.stepcount = 0
        self.stepcost = 0.
        self.steptimer = 0.
        
        self.index = T.lscalar('batch_index_in_sgd') 
        self.incs = dict([(
            p, 
            theano.shared(value=numpy.zeros(p.get_value().shape, 
                                            dtype=theano.config.floatX),
                          name='inc_' + p.name,
                          broadcastable=p.broadcastable)
        ) for p in self.params])

        self.grad = T.grad(self.cost, self.params)
        # set "gradient"
        for ilayer in model.models_stack:
            if hasattr(ilayer, 'quantized_bprop'):
                ilayer.quantized_bprop(self.cost)
                i = 0
                for iparam in self.params:
                    if ilayer.w is iparam:
                        self.grad[i] = ilayer.dEdW
                    i += 1
        print self.grad
        print self.params
        # /set "gradient"

        self.set_learningrate(learningrate)
        
        params_vector = T.concatenate([p.flatten() for p in self.params])
        self.get_params_value = theano.function([], params_vector)
        self.ref_vector = theano.shared(value=self.get_params_value(),
                                        name='step_ref_params',
                                        borrow=True)

        inc_vector = params_vector - self.ref_vector
        norm_ref_vector = T.sqrt(T.sum(self.ref_vector ** 2))
        norm_inc_vector = T.sqrt(T.sum(inc_vector ** 2))
        angle_rad = T.arccos(T.dot(self.ref_vector, inc_vector) /\
                    (norm_ref_vector * norm_inc_vector))

        self.get_step_info = theano.function([], (norm_inc_vector, angle_rad))

    def set_learningrate(self, learningrate):
        self.learningrate  = learningrate
        self.inc_updates = []  # updates the parameter increasements (i.e. 
                               # value in the self.incs dictionary.). Due to 
                               # momentum, the increasement itself is
                               # changing between epochs. Its increasing by:
                               # from (key) inc_params 
                               # to (value) momentum * inc_params - lr * grad
                               
        self.updates = []  # updates the parameters of model during each epoch.
                           # from (key) params
                           # to (value) params + inc_params
                           
        for _param, _grad in zip(self.params, self.grad):
            self.inc_updates.append(
                (self.incs[_param],
                 self.momentum * self.incs[_param] - self.learningrate * _grad
                )
            )
            self.updates.append((_param, _param + self.incs[_param]))

        if not self.supervised:
            self._updateincs = theano.function(
                inputs = [self.index], 
                outputs = self.cost, 
                updates = self.inc_updates,
                givens = {
                    self.varin : self.data[self.index * self.batchsize: \
                                           (self.index+1)*self.batchsize]
                }
            )
        else:
            self._updateincs = theano.function(
                inputs = [self.index],
                outputs = self.cost,
                updates = self.inc_updates,
                givens = {
                    self.varin : self.data[self.index * self.batchsize: \
                                           (self.index+1)*self.batchsize],
                    self.truth : self.truth_data[self.index * self.batchsize: \
                                                 (self.index+1)*self.batchsize]
                }
            )

        self._trainmodel = theano.function(inputs=[], updates = self.updates)

    def epoch(self):
        start = time.time()
        stepcount = 0.0
        cost = 0.
        self.ref_vector.set_value(self.get_params_value())
        for batch_index in self.rng.permutation(self.numbatches - 1):
            stepcount += 1.0
            # This is Roland's way of computing cost, still mean over all
            # batches. It saves space and don't harm computing time... 
            # But a little bit unfamilliar to understand at first glance.
            cost = (1.0 - 1.0/stepcount) * cost + \
                   (1.0/stepcount) * self._updateincs(batch_index)
            
            self._trainmodel()

        norm, angle_rad = self.get_step_info()
        self.epochcount += 1
        stop = time.time()
        if self.verbose:
            print 'epoch %d: %.2fs, lr %.3g cost %.6g, ' % (
                self.epochcount, (stop - start), self.learningrate, cost) + \
                  'update norm %.3g angle(RAD) %.3f' % (norm, angle_rad)

        return cost

    def step(self, verbose_stride=1):
        """
        Randomly pick a minibatch from dataset, and perform one step of update.
        
        If you switch this method between self.epoch() during training, the
        update norm, angle may not be immediately correct after the epoch/step
        at which you switch.
        """
        start = time.time()
        self.ref_vector.set_value(self.get_params_value())
        batch_index = self.rng.randint(0, self.numbatches - 1)
        
        cost = self._updateincs(batch_index)
        self._trainmodel()
        
        self.stepcost += cost
        self.stepcount += 1

        norm, angle_rad = self.get_step_info()
        stop = time.time()
        self.steptimer += (stop - start)
        if (self.stepcount % verbose_stride == 0) and self.verbose:
            print 'minibatch %d: %.2fs, lr %.3g cost %.6g, ' % (
                self.stepcount, self.steptimer, self.learningrate,
                self.stepcost / verbose_stride) + \
                  'update norm %.3g angle(RAD) %.3f' % (norm, angle_rad)
            self.steptimer = 0.
            self.stepcost = 0.
        return cost

    def step_fast(self, verbose_stride=1):
        """
        A faster implementation of step(). Removes evaluation of angles,
        norms, etc.
        
        MUCH FASTER!! ~ 20 times!!
        """
        start = time.time()
        batch_index = self.rng.randint(0, self.numbatches - 1)
        
        cost = self._updateincs(batch_index)
        self._trainmodel()
        
        self.stepcost += cost
        self.stepcount += 1

        stop = time.time()
        self.steptimer += (stop - start)
        if (self.stepcount % verbose_stride == 0) and self.verbose:
            print 'minibatch %d: %.2fs, lr %.3g cost %.6g ' % (
                self.stepcount, self.steptimer, self.learningrate,
                self.stepcost / verbose_stride)
            self.steptimer = 0.
            self.stepcost = 0.
        return cost


class Dropout(object):
    class DropoutLayer(Layer):
        def __init__(self, n_in, droprate, varin=None, theano_rng=None):
            super(Dropout.DropoutLayer, self).__init__(n_in, n_in, varin=varin)
            assert (droprate >= 0. and droprate <= 1.), \
                "droprate has to be in the interval [0, 1]."
            self.droprate = droprate

            if not theano_rng:
                theano_rng = RandomStreams(123)
            assert isinstance(theano_rng, T.shared_randomstreams.RandomStreams)
            self.theano_rng = theano_rng

        def output(self):
            return self.theano_rng.binomial(size=self.varin.shape, n=1,
                p = 1 - self.droprate,
                dtype=theano.config.floatX
            ) * self.varin

        def _print_str(self):
            return "    (" + self.__class__.__name__ + ": droprate " + \
                str(self.droprate) + ")"
            
    def __init__(self, model, droprates, theano_rng=None):
        """
        Build a noisy model according to the passed model, which is an
        Autoencoder object or StackedLayer object. The newly built model shares
        the same theano shared parameters with the initial one, but with
        binomial zero-one noise injected at each layer. The ratios of dropped
        units in each layer is spectfied at droprates, which is a list starting
        from input layer.
        """
        if not theano_rng:
            theano_rng = RandomStreams(123)
        assert isinstance(theano_rng, T.shared_randomstreams.RandomStreams)
        self.theano_rng = theano_rng
        self.model = model 
        self.set_droprates(droprates)

    def set_droprates(self, droprates):
        self.droprates = droprates
        if isinstance(self.model, AutoEncoder):
            assert len(droprates) == 2, "List \"droprates\" has a wrong length."
            self.dropout_model = copy.copy(self.model)
            def dropout_encoder():
                return self.DropoutLayer(
                    n_in=self.dropout_model.n_in,
                    droprate=self.droprates[0],
                    varin=self.model.varin,
                    theano_rng=self.theano_rng
                ) + self.model.encoder()
            self.dropout_model.encoder = dropout_encoder
            def dropout_decoder():
                return self.DropoutLayer(
                    n_in=self.dropout_model.n_hid,
                    droprate=self.droprates[1],
                    varin=self.dropout_model.encoder().output(),
                    theano_rng=self.theano_rng
                ) + self.model.decoder()
            self.dropout_model.decoder = dropout_decoder

        elif isinstance(self.model, StackedLayer):
            # TODO: more thing to do for assertion here. Not sure if it will
            # work for nested StackedLayer object.
            # assert len(droprates) == len(self.model.models_stack), \
            #        "List \"droprates\" has a wrong length."
            self.dropout_model = None
            i = 0
            for layer_model in self.model.models_stack:
                copied_layer = copy.copy(layer_model)
                if layer_model.params != []:
                    combination = self.DropoutLayer(
                        n_in=layer_model.n_in,
                        droprate=self.droprates[i],
                        theano_rng=self.theano_rng
                    ) + copied_layer
                    if self.dropout_model == None:
                        self.dropout_model = combination
                    else:
                        self.dropout_model = self.dropout_model + combination
                    i += 1
                else:
                    if self.dropout_model == None:
                        self.dropout_model = copied_layer
                    else:
                        self.dropout_model = self.dropout_model + copied_layer

        elif isinstance(self.model, Layer):
            assert len(droprates) == 1, "List \"droprates\" has a wrong length."
            if self.model.params != []:
                self.dropout_model = self.DropoutLayer(
                    n_in=self.model.n_in,
                    droprate=self.droprates[0],
                    theano_rng=self.theano_rng
                ) + copy.copy(self.model)
            else:
                raise TypeError("Dropout on a layer with no parameters has " + \
                                "no meaning currently")
        else:
            raise TypeError("Passed model has to be an Autoencoder, " + \
                            "StackedLayer or single Layer")


class BatchNormalization(object):
    def __init__(self, model, npy_rng=None):
        """
        TODO: check if it works for autoencoders. Use moving average instead to
        allow for real normalization at test time.
        
        A batch normalization implementation according to the following paper:

        http://arxiv.org/pdf/1502.03167.pdf
        
        Build a model with batch normalization according to the passed model,
        which could be an AutoEncoder object or StackedLayer object. The newly
        built model shares the same theano shared parameters with the initial
        one, but has inplanted batch normalization at PRE-activations (i.e.
        before applying the activation function). Also, it establishes a new
        parameter group: params_batchnorm. In addition to all those parameters
        contained in params, params_batchnorm has included [gamma, beta], which
        are introduced by batch normalizaion.

        Remember to apply this class at the last step of any other tricks, like
        dropout etc.

        At test time, the means and variances are over the whole dataset. So it
        should be a big batch containing all the training/testing samples for
        this implementation. Will use an moving average instead in future
        updates. 
        """
        if npy_rng is None:
            npy_rng = numpy.random.RandomState(3456)
        assert isinstance(npy_rng, numpy.random.RandomState), \
            "npy_rng has to be a random number generater."
        self.npy_rng = npy_rng
        self.model = model 

        if isinstance(self.model, AutoEncoder):
            self.dropout_model = copy.copy(self.model)
            def batchnorm_encoder():
                encoder_layer = self.model.encoder()
                assert not isinstance(encoder_layer, StackedLayer), (
                    "Batch normalization on deep autoencoder with more than "
                    "1 layer of encoder/decoder is not supported.")
                encoder_layer.gamma = theano.shared(
                    numpy.ones(self.model.n_out, dtype=theano.config.floatX),
                    name='gamma_encoder', borrow=True
                )
                encoder_layer.beta = theano.shared(
                    numpy.zeros(self.model.n_out, dtype=theano.config.floatX),
                    name='beta_encoder', borrow=True
                )
                encoder_layer.params_batchnorm = encoder_layer.params + \
                    [encoder_layer.gamma, encoder_layer.beta]
                def normed_fanin_encoder():
                    return (
                        self.model.encoder().fanin() - \
                        self.model.encoder().fanin().mean(
                            axis=0, keepdims=True)
                    ) / (
                        self.model.encoder().fanin().std(
                            axis=0, keepdims=True
                        ) + 1E-6
                    ) * encoder_layer.gamma + encoder_layer.beta
                encoder_layer.fanin = normed_fanin_encoder
                return encoder_layer
            self.batchnorm_model.encoder = batchnorm_encoder
            
        elif isinstance(self.model, StackedLayer):
            # TODO: more thing to do for assertion here. Not sure if it will
            # work for nested StackedLayer object.
            # assert len(droprates) == len(self.model.models_stack), \
            #        "List \"droprates\" has a wrong length."
            copied_model_list = []
            for layer_model in self.model.models_stack:
                copied_model_list.append(copy.copy(layer_model))
            copied_model = StackedLayer(models_stack=copied_model_list,
                                        varin=self.model.varin)
            
            copied_model.params_batchnorm = []
            prev_layer = None
            for layer_model in copied_model.models_stack:
                # process prev_layer
                if prev_layer != None and prev_layer.params != []:
                    prev_layer.gamma = theano.shared(
                        numpy.ones(prev_layer.n_out,
                                   dtype=theano.config.floatX),
                        name='gamma', borrow=True
                    )
                    prev_layer.beta = theano.shared(
                        numpy.zeros(prev_layer.n_out,
                                    dtype=theano.config.floatX),
                        name='beta', borrow=True
                    )
                    prev_layer.params_batchnorm = \
                        prev_layer.params + [prev_layer.gamma, prev_layer.beta]
                    copied_model.params_batchnorm += \
                        prev_layer.params_batchnorm
                    
                    wx = prev_layer.fanin()  # wx stands for w * x.
                    layer_model.varin = prev_layer.output(
                        (
                            wx - wx.mean(axis=0, keepdims=True)
                        ) / (
                            wx.std(axis=0, keepdims=True) + 1E-6
                        ) * prev_layer.gamma + prev_layer.beta
                    )
                prev_layer = layer_model
            
            # process the last layer
            if layer_model.params != []:
                layer_model.gamma = theano.shared(
                    numpy.ones(layer_model.n_out,
                               dtype=theano.config.floatX),
                    name='gamma', borrow=True
                )
                layer_model.beta = theano.shared(
                    numpy.zeros(layer_model.n_out,
                                dtype=theano.config.floatX),
                    name='beta', borrow=True
                )
                layer_model.params_batchnorm = \
                    layer_model.params + [layer_model.gamma, layer_model.beta]
                copied_model.params_batchnorm += layer_model.params_batchnorm

                def last_layer_output(fanin=None):
                    wx = layer_model.fanin()  # wx stands for w * x.
                    return self.model.models_stack[-1].output(
                        (
                            wx - wx.mean(axis=0, keepdims=True)
                        ) / (
                            wx.std(axis=0, keepdims=True) + 1E-6
                        ) * layer_model.gamma + layer_model.beta
                    )
                layer_model.output = last_layer_output
            
            self.batchnorm_model = copied_model

        elif isinstance(self.model, Layer):
            if self.model.params != []:
                copied_layer = copy.copy(self.model)
                copied_layer.gamma = theano.shared(
                    numpy.ones(layer_model.n_in,
                               dtype=theano.config.floatX),
                    name='gamma', borrow=True
                )
                copied_layer.beta = theano.shared(
                    numpy.zeros(layer_model.n_in,
                                dtype=theano.config.floatX),
                    name='beta', borrow=True
                )
                copied_layer.params_batchnorm = layer_model.params + \
                    [copied_layer.gamma, copied_layer.beta]
                def normed_fanin():
                    return (
                        self.model.fanin() - self.model.fanin().mean(
                            axis=0, keepdims=True)
                    ) / (
                        self.model.fanin().std(axis=0, keepdims=True) + 1E-6
                    ) * copied_layer.gamma + copied_layer.beta
                copied_layer.fanin = normed_fanin
                self.batchnorm_model = copied_layer
            else:
                raise TypeError("Batch normalization on a layer with no "
                                "parameters has no meaning currently.")
        else:
            raise TypeError("Passed model has to be an Autoencoder, "
                            "StackedLayer or Layer object.")


