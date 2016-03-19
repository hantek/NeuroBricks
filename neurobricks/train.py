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
            theano.shared(value=numpy.zeros(p.shape.eval(), 
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

        delta_vector = params_vector - self.ref_vector
        norm_ref_vector = T.sqrt(T.sum(self.ref_vector ** 2))
        norm_delta_vector = T.sqrt(T.sum(delta_vector ** 2))
        angle_rad = T.arccos(T.dot(self.ref_vector, delta_vector) /\
                    (norm_ref_vector * norm_delta_vector))

        self.get_step_info = theano.function([], (norm_delta_vector, angle_rad))

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


class Adam(object):
    def __init__(self, varin, data, cost, params, 
                 truth=None, truth_data=None, supervised=False,
                 batchsize=100, learningrate=0.1, beta_1=0.9, beta_2=0.999,
                 rng=None, verbose=True):
        """
        Adam learning rule, an implelentation for paper:
            http://arxiv.org/abs/1412.6980
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
        self.beta_1        = beta_1
        self.beta_2        = beta_2
        self.beta_1_pow    = theano.shared(
                             numpy.asarray(1., dtype=theano.config.floatX))
        self.beta_2_pow    = theano.shared(
                             numpy.asarray(1., dtype=theano.config.floatX))
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
        self.m = dict([(
            p, 
            theano.shared(
                value=numpy.zeros(p.shape.eval(), dtype=theano.config.floatX),
                name=p.name + '_m',
                broadcastable=p.broadcastable
            )
        ) for p in self.params])

        self.v = dict([(
            p, 
            theano.shared(
                value=numpy.zeros(p.shape.eval(), dtype=theano.config.floatX),
                name=p.name + '_v',
                broadcastable=p.broadcastable
            )
        ) for p in self.params])

        self.grad = T.grad(self.cost, self.params)

        self.set_learningrate(learningrate)
        
        params_vector = T.concatenate([p.flatten() for p in self.params])
        self.get_params_value = theano.function([], params_vector)
        self.ref_vector = theano.shared(value=self.get_params_value(),
                                        name='step_ref_params',
                                        borrow=True)

        delta_vector = params_vector - self.ref_vector
        norm_ref_vector = T.sqrt(T.sum(self.ref_vector ** 2))
        norm_delta_vector = T.sqrt(T.sum(delta_vector ** 2))
        angle_rad = T.arccos(T.dot(self.ref_vector, delta_vector) /\
                    (norm_ref_vector * norm_delta_vector))

        self.get_step_info = theano.function([], (norm_delta_vector, angle_rad))

    def set_learningrate(self, learningrate):
        self.learningrate  = learningrate
        adjusted_lr = self.learningrate * T.sqrt(1. - self.beta_2_pow) \
                      / (1. - self.beta_1_pow) 
        self.update_beta_pows = [
            (self.beta_1_pow, self.beta_1_pow * self.beta_1),
            (self.beta_2_pow, self.beta_2_pow * self.beta_2)]

        self.update_m = []
        self.update_v = []
        self.update_params = []
        for _param, _grad in zip(self.params, self.grad):
            self.update_m.append((
                self.m[_param],
                self.beta_1 * self.m[_param] + (1 - self.beta_1) * _grad
            ))
            self.update_v.append((
                self.v[_param],
                self.beta_2 * self.v[_param] + (1 - self.beta_2) * _grad**2
            ))
            self.update_params.append((
                _param,
                _param - adjusted_lr * self.m[_param] \
                         / (T.sqrt(self.v[_param]) + 1e-8)
            ))

        if not self.supervised:
            self.fun_update_mv_betapows = theano.function(
                inputs=[self.index],
                outputs=self.cost,
                updates=self.update_m + self.update_v + self.update_beta_pows,
                givens={
                    self.varin : self.data[self.index * self.batchsize: \
                                           (self.index+1)*self.batchsize]
                }
            )
        else:
            self.fun_update_mv_betapows = theano.function(
                inputs=[self.index],
                outputs=self.cost,
                updates=self.update_m + self.update_v + self.update_beta_pows,
                givens={
                    self.varin : self.data[self.index * self.batchsize: \
                                           (self.index+1)*self.batchsize],
                    self.truth : self.truth_data[self.index * self.batchsize: \
                                                 (self.index+1)*self.batchsize]
                }
            )

        self._trainmodel = theano.function(inputs=[],
                                           updates=self.update_params)

    def epoch(self):
        start = time.time()
        stepcount = 0.0
        cost = 0.
        self.ref_vector.set_value(self.get_params_value())
        for batch_index in self.rng.permutation(self.numbatches - 1):
            stepcount += 1.0
            cost = (1.0 - 1.0/stepcount) * cost + \
                   (1.0/stepcount) * self.fun_update_mv_betapows(batch_index)
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
        
        cost = self.fun_update_mv_betapows(batch_index)
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
        
        cost = self.fun_update_mv_betapows(batch_index)
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
            theano.shared(value=numpy.zeros(p.shape.eval(), 
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
        # /set "gradient"

        self.set_learningrate(learningrate)
        
        params_vector = T.concatenate([p.flatten() for p in self.params])
        self.get_params_value = theano.function([], params_vector)
        self.ref_vector = theano.shared(value=self.get_params_value(),
                                        name='step_ref_params',
                                        borrow=True)

        delta_vector = params_vector - self.ref_vector
        norm_ref_vector = T.sqrt(T.sum(self.ref_vector ** 2))
        norm_delta_vector = T.sqrt(T.sum(delta_vector ** 2))
        angle_rad = T.arccos(T.dot(self.ref_vector, delta_vector) /\
                    (norm_ref_vector * norm_delta_vector))

        self.get_step_info = theano.function([], (norm_delta_vector, angle_rad))

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
    def __init__(self, model, BN_params=None, BN_meanstds=None, npy_rng=None):
        """
        TODO: Not working ideally for autoencoders and single layers.
        
        A batch normalization implementation according to the following paper:

        http://arxiv.org/pdf/1502.03167.pdf
        
        Basically it bulids 2 models: self.batchnorm_model, and 
        self.batchnorm_test. The former one is for training, while the latter
        is only for test time use. Note that the self.batchnorm_test model is
        not completed, that is to say, it is calling some of the attribute
        members from self.batchnorm_model. As the two models are binded in a
        same object, this can always be safe.

        self.batchnorm_model is built according to the passed model,
        which could be an AutoEncoder object or StackedLayer object. It
        shares the same theano shared parameters with the initial
        one, but has inplanted batch normalization at PRE-activations (i.e.
        before applying the activation function).
        
        Also, the class provides 2 new parameter groups:
         - params_batchnorm. Includes [gamma, beta], which are introduced by
           batch normalizaion.
         - params_meanstds. Includes [wxmean, wxstd], which is the mean and standard
           deviation of each layer's representation. It is only used by
           self.batchnorm_test model, and irrelevant to the training process.
           You will also need to update it before each time of test. 

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
        if BN_params:
            self.BN_params = BN_params[::-1]
        else:
            self.BN_params = None

        if BN_meanstds:
            self.BN_meanstds = BN_meanstds[::-1]
        else:
            self.BN_meanstds = None

        if isinstance(self.model, AutoEncoder):
            print "WARNING: May not be correct now."
            self.batchnorm_model = copy.copy(self.model)
            def batchnorm_encoder():
                encoder_layer = self.model.encoder()
                assert not isinstance(encoder_layer, StackedLayer), (
                    "Batch normalization on deep autoencoder with more than "
                    "1 layer of encoder/decoder is not supported.")
                if not self.BN_params:
                    encoder_layer.gamma = theano.shared(
                        numpy.ones(self.model.n_out,
                                   dtype=theano.config.floatX),
                        name='gamma_encoder', borrow=True
                    )
                    encoder_layer.beta = theano.shared(
                        numpy.zeros(self.model.n_out,
                                    dtype=theano.config.floatX),
                        name='beta_encoder', borrow=True
                    )
                else:
                    encoder_layer.gamma = self.BN_params.pop()
                    encoder_layer.beta = self.BN_params.pop()
                encoder_layer.params_batchnorm = encoder_layer.params + \
                    [encoder_layer.gamma, encoder_layer.beta]
                encoder_layer.params_extra = \
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
            """
            NOTE: if dealing with convolutional nets, there is one mean value
            over each feature map, not each location! So the number (and shape)
            of means and stds should be equal to biases.
            """
            # TODO: more thing to do for assertion here. Not sure if it will
            # work for nested StackedLayer object.
            # assert len(droprates) == len(self.model.models_stack), \
            #        "List \"droprates\" has a wrong length."
            copied_model_list = []
            copied_model_list2 = []
            for layer_model in self.model.models_stack:
                copied_model_list.append(copy.copy(layer_model))
                copied_model_list2.append(copy.copy(layer_model))
            copied_model = StackedLayer(models_stack=copied_model_list,
                                        varin=self.model.varin)
            copied_model_test = StackedLayer(models_stack=copied_model_list2,
                                             varin=self.model.varin)

            self.params_batchnorm = []
            self.params_meanstds = []
            prev_layer = None
            for (layer_model, layer_model_test) in zip(
                copied_model.models_stack, copied_model_test.models_stack):
                # process prev_layer
                if prev_layer != None and prev_layer.params != []:
                    if not self.BN_params:
                        shape = prev_layer.n_out
                        if hasattr(prev_layer, 'conv'):
                            shape = (1, shape[1], 1, 1)
                            broadcastable = (True, False, True, True)
                        else:
                            shape = (1, shape)
                            broadcastable = (True, False)

                        print "Adding Batch Normalization to the layer \t" + \
                              prev_layer.__class__.__name__ + " " + \
                              str(prev_layer.n_out)

                        prev_layer.gamma = theano.shared(
                            numpy.ones(shape, dtype=theano.config.floatX),
                            name=prev_layer.__class__.__name__ + '_gamma',
                            borrow=True,
                            broadcastable=broadcastable
                        )
                        prev_layer.beta = theano.shared(
                            numpy.zeros(shape, dtype=theano.config.floatX),
                            name=prev_layer.__class__.__name__ + '_beta',
                            borrow=True,
                            broadcastable=broadcastable
                        )
                        prev_layer.wxmean = theano.shared(
                            numpy.zeros(shape, dtype=theano.config.floatX),
                            name = prev_layer.__class__.__name__ + '_wxmean',
                            borrow=True,
                            broadcastable=broadcastable
                        )
                        prev_layer.wxstd = theano.shared(
                            numpy.zeros(shape, dtype=theano.config.floatX),
                            name = prev_layer.__class__.__name__ + '_wxstd',
                            borrow=True,
                            broadcastable=broadcastable
                        )
                    else:
                        prev_layer.gamma = self.BN_params.pop()
                        prev_layer.beta = self.BN_params.pop()
                        prev_layer.wxmean = self.BN_meanstds.pop()
                        prev_layer.wxstd = self.BN_meanstds.pop()
                    
                    self.params_batchnorm += [prev_layer.gamma, prev_layer.beta]
                    self.params_meanstds += [prev_layer.wxmean,
                                             prev_layer.wxstd]
                    
                    wx = prev_layer.fanin()  # wx stands for w * x.
                    if hasattr(prev_layer, 'conv'):
                        next_varin = prev_layer.output(
                            (
                                wx - wx.mean(axis=(0, 2, 3), keepdims=True)
                            ) / (
                                wx.std(axis=(0, 2, 3), keepdims=True) + 1E-6
                            ) * prev_layer.gamma + prev_layer.beta
                        )
                    else:
                        next_varin = prev_layer.output(
                            (
                                wx - wx.mean(axis=0, keepdims=True)
                            ) / (
                                wx.std(axis=0, keepdims=True) + 1E-6
                            ) * prev_layer.gamma + prev_layer.beta
                        )
                    wx_test = prev_layer_test.fanin()
                    next_varin_test = prev_layer_test.output(
                        (wx_test - prev_layer.wxmean) / (prev_layer.wxstd + 1E-6
                        ) * prev_layer.gamma + prev_layer.beta
                    )

                elif prev_layer != None:
                    next_varin = prev_layer.output()
                    next_varin_test = prev_layer_test.output()

                if prev_layer != None:
                    right_is_conv = hasattr(layer_model, 'conv')
                    left_is_conv = hasattr(prev_layer, 'conv')
                    if (left_is_conv       and right_is_conv) or (
                        (not left_is_conv) and (not right_is_conv)):
                        layer_model.varin = next_varin
                        layer_model_test.varin = next_varin_test
                    elif not right_is_conv:
                        layer_model.varin = next_varin.flatten(2)
                        layer_model_test.varin = next_varin_test.flatten(2)
                    elif not left_is_conv:  # FC-CONV
                        layer_model.varin = next_varin.reshape(
                            layer_model.n_in)
                        layer_model_test.varin = next_varin_test.reshape(
                            layer_model.n_in)
                
                prev_layer = layer_model
                prev_layer_test = layer_model_test
            
            # process the last layer
            if layer_model.params != []:
                if not self.BN_params:
                    shape = layer_model.n_out
                    if hasattr(layer_model, 'conv'):
                        shape = (1, shape[1], 1, 1)
                        broadcastable = (True, False, True, True)
                    else:
                        shape = (1, shape)
                        broadcastable = (True, False)
                    
                    print "Adding Batch Normalization to the layer \t" + \
                          prev_layer.__class__.__name__ + " " + \
                          str(prev_layer.n_out)

                    layer_model.gamma = theano.shared(
                        numpy.ones(shape, dtype=theano.config.floatX),
                        name=layer_model.__class__.__name__ + '_gamma',
                        borrow=True,
                        broadcastable=broadcastable
                    )
                    layer_model.beta = theano.shared(
                        numpy.zeros(shape, dtype=theano.config.floatX),
                        name=layer_model.__class__.__name__ + '_beta',
                        borrow=True,
                        broadcastable=broadcastable
                    )
                    layer_model.wxmean = theano.shared(
                        numpy.zeros(shape, dtype=theano.config.floatX),
                        name=layer_model.__class__.__name__ + '_wxmean',
                        borrow=True,
                        broadcastable=broadcastable
                    )
                    layer_model.wxstd = theano.shared(
                        numpy.zeros(shape, dtype=theano.config.floatX),
                        name=layer_model.__class__.__name__ + '_wxstd',
                        borrow=True,
                        broadcastable=broadcastable
                    )
                else:
                    layer_model.gamma = self.BN_params.pop()
                    layer_model.beta = self.BN_params.pop()
                    layer_model.wxmean = self.BN_meanstds.pop()
                    layer_model.wxstd = self.BN_meanstds.pop()
                    
                self.params_batchnorm += [layer_model.gamma, layer_model.beta]
                self.params_meanstds += [layer_model.wxmean, layer_model.wxstd]
                    
                if hasattr(layer_model, 'conv'):
                    def last_layer_output(fanin=None):
                        wx = layer_model.fanin()  # wx stands for w * x.
                        return self.model.models_stack[-1].output(
                            (
                                wx - wx.mean(axis=(0, 2, 3), keepdims=True)
                            ) / (
                                wx.std(axis=(0, 2, 3), keepdims=True) + 1E-6
                            ) * layer_model.gamma + layer_model.beta
                        )
                else:
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
            
                def last_layer_output_test(fanin=None):
                    wx = layer_model.fanin()  # wx stands for w * x.
                    return self.model.models_stack[-1].output(
                        (wx - layer_model.wxmean) / (layer_model.wxstd + 1E-6
                        ) * layer_model.gamma + layer_model.beta
                    )
                layer_model.output = last_layer_output
                layer_model_test.output = last_layer_output_test
            
            self.batchnorm_model = copied_model
            self.batchnorm_test = copied_model_test

        elif isinstance(self.model, Layer):
            print "WARNING: May not be correct now."
            if self.model.params != []:
                copied_layer = copy.copy(self.model)
                if not self.BN_params:
                    shape = copied_layer.n_out
                    broadcastable = (False, )
                    if hasattr(copied_layer, 'conv'):
                        shape = (1,) + shape[1:]
                        broadcastable = (True,) + (False, ) * (len(shape) - 1)

                    copied_layer.gamma = theano.shared(
                        numpy.ones(shape, dtype=theano.config.floatX),
                        name=copied_layer.__class__.__name__ + '_gamma',
                        borrow=True,
                        broadcastable=broadcastable
                    )
                    copied_layer.beta = theano.shared(
                        numpy.zeros(shape, dtype=theano.config.floatX),
                        name=copied_layer.__class__.__name__ + '_beta',
                        borrow=True,
                        broadcastable=broadcastable
                    )
                else:
                    copied_layer.gamma = self.BN_params.pop()
                    copied_layer.beta = self.BN_params.pop()
                copied_layer.params_batchnorm = layer_model.params + \
                    [copied_layer.gamma, copied_layer.beta]
                copied_layer.params_extra = \
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

    def compute_meanstds(self, given_data):
        if not hasattr(self, 'theano_funcs'):
            #these are for computing means and stds at each layer.
            self.theano_funcs = []
            for layer in self.batchnorm_model.models_stack:
                if layer.params != []:
                    if hasattr(layer, 'conv'):
                        self.theano_funcs.append(theano.function(
                            [self.batchnorm_model.varin], 
                            layer.varfanin.mean(axis=(0, 2, 3), keepdims=True),
                        ))
                        self.theano_funcs.append(theano.function(
                            [self.batchnorm_model.varin],
                            layer.varfanin.std(axis=(0, 2, 3), keepdims=True),
                        ))
                    else:
                        self.theano_funcs.append(theano.function(
                            [self.batchnorm_model.varin], 
                            layer.varfanin.mean(axis=0, keepdims=True),
                        ))
                        self.theano_funcs.append(theano.function(
                            [self.batchnorm_model.varin],
                            layer.varfanin.std(axis=0, keepdims=True),
                        ))

        return [ifunc(given_data) for ifunc in self.theano_funcs]
    
    def set_mean_stds(self, meanstds):
        """
        set the mean and stds at each layer according to the passed input.
        """
        assert len(meanstds) == len(self.params_meanstds), (
            "passed list length is not consistent with the model's.")
        for ishared, idata in zip(self.params_meanstds, meanstds):
            ishared.set_value(idata)
