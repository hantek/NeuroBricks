import os
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import cPickle

from dataset import CIFAR10
from classifier import LogisticRegression
from model import ReluAutoencoder
from preprocess import SubtractMeanAndNormalizeH, PCA
from train import GraddescentMinibatch, Adam
from params import save_params, load_params, set_params, get_params

import pdb


#######################
# SET SUPER PARAMETER #
#######################

pca_retain = 800
hid_layer_sizes = [4000, 4000]
batchsize = 100

momentum = 0.9
beta_1 = 0.9
beta_2 = 0.999

lr = 5e-3
epc = 1000

print " "
print "pca_retain =", pca_retain
print "hid_layer_sizes =", hid_layer_sizes
print "batchsize =", batchsize
print "momentum =", momentum
print "beta_1 = %f, beta2 = %f" % (beta_1, beta_2)
print "lr = %f, epc = %d" % (lr, epc)

#############
# LOAD DATA #
#############

cifar10_data = CIFAR10()
train_x, train_y = cifar10_data.get_train_set()
test_x, test_y = cifar10_data.get_test_set()

print "\n... pre-processing"
preprocess_model = SubtractMeanAndNormalizeH(train_x.shape[1])
map_fun = theano.function([preprocess_model.varin], preprocess_model.output())

pca_obj = PCA()
pca_obj.fit(map_fun(train_x), retain=pca_retain, whiten=True)
preprocess_model = preprocess_model + pca_obj.forward_layer
preprocess_function = theano.function([preprocess_model.varin], preprocess_model.output())
train_x = preprocess_function(train_x)
test_x = preprocess_function(test_x)

feature_num = train_x.shape[0] * train_x.shape[1]

train_x = theano.shared(value=train_x, name='train_x', borrow=True)
train_y = theano.shared(value=train_y, name='train_y', borrow=True)
test_x = theano.shared(value=test_x, name='test_x', borrow=True)
test_y = theano.shared(value=test_y, name='test_y', borrow=True)
print "Done."

#########################
# BUILD PRE-TRAIN MODEL #
#########################

print "... building models"
npy_rng = numpy.random.RandomState(123)
model = ReluAutoencoder(
    train_x.get_value().shape[1], hid_layer_sizes[0], 
    init_w = theano.shared(
        value=0.01 * train_x.get_value()[:hid_layer_sizes[0], :].T,
        name='w_0',
        borrow=True
    ),
    vistype='real', tie=True, npy_rng=npy_rng
) + ReluAutoencoder(
    hid_layer_sizes[0], hid_layer_sizes[1],
    init_w = theano.shared(
        value=numpy.tile(
            0.01 * train_x.get_value(),
            (hid_layer_sizes[0] * hid_layer_sizes[1] / feature_num + 1, 1)
        ).flatten()[:(hid_layer_sizes[0] * hid_layer_sizes[1])].reshape(
            hid_layer_sizes[0], hid_layer_sizes[1]
        ),
        name='w_2',
        borrow=True
    ),
    vistype='real', tie=True, npy_rng=npy_rng
) + LogisticRegression(
    hid_layer_sizes[1], 10, npy_rng=npy_rng
)

print "\nModel for SGD:"
model.print_layer()

train_set_error_rate = theano.function(
    [],
    T.mean(T.neq(model.models_stack[-1].predict(), train_y)),
    givens = {model.varin : train_x},
)
test_set_error_rate = theano.function(
    [],
    T.mean(T.neq(model.models_stack[-1].predict(), test_y)),
    givens = {model.varin : test_x},
)

###############################################################################
# second model, for adam algo
###############################################################################

model_adam = ReluAutoencoder(
    train_x.get_value().shape[1], hid_layer_sizes[0], 
    init_w = theano.shared(
        value=0.01 * train_x.get_value()[:hid_layer_sizes[0], :].T,
        name='w_0_adam',
        borrow=True
    ),
    vistype='real', tie=True, npy_rng=npy_rng
) + ReluAutoencoder(
    hid_layer_sizes[0], hid_layer_sizes[1],
    init_w = theano.shared(
        value=numpy.tile(
            0.01 * train_x.get_value(),
            (hid_layer_sizes[0] * hid_layer_sizes[1] / feature_num + 1, 1)
        ).flatten()[:(hid_layer_sizes[0] * hid_layer_sizes[1])].reshape(
            hid_layer_sizes[0], hid_layer_sizes[1]
        ),
        name='w_2_adam',
        borrow=True
    ),
    vistype='real', tie=True, npy_rng=npy_rng
) + LogisticRegression(
    hid_layer_sizes[1], 10, npy_rng=npy_rng
)

print "\nModel for Adam:"
model_adam.print_layer()

train_set_error_rate_adam = theano.function(
    [],
    T.mean(T.neq(model_adam.models_stack[-1].predict(), train_y)),
    givens = {model_adam.varin : train_x},
)
test_set_error_rate_adam = theano.function(
    [],
    T.mean(T.neq(model_adam.models_stack[-1].predict(), test_y)),
    givens = {model_adam.varin : test_x},
)
print "Done."

#############
# FINE-TUNE #
#############

print "\n\n... fine-tuning the whole network"
trainer = GraddescentMinibatch(
    varin=model.varin, data=train_x, 
    truth=model.models_stack[-1].vartruth, truth_data=train_y,
    supervised=True,
    cost=model.models_stack[-1].cost(), 
    params=model.params,
    batchsize=batchsize, learningrate=lr, momentum=momentum,
    rng=npy_rng
)

trainer_adam = Adam(
    varin=model_adam.varin, data=train_x, 
    truth=model_adam.models_stack[-1].vartruth, truth_data=train_y,
    supervised=True,
    cost=model_adam.models_stack[-1].cost(), 
    params=model_adam.params,
    batchsize=batchsize, learningrate=10 * lr, beta_1=beta_1, beta_2=beta_2,
    rng=npy_rng

)

prev_cost = numpy.inf
prev_cost_adam = numpy.inf
for epoch in xrange(epc):
    print " "
    cost = trainer.epoch()
    print "SGD error rate, train: %f, test: %f" % (
        train_set_error_rate(), test_set_error_rate()
    )
    cost_adam = trainer_adam.epoch()
    print "Adam error rate, train: %f, test: %f" % (
        train_set_error_rate_adam(), test_set_error_rate_adam()
    )

    if prev_cost <= cost:
        trainer.set_learningrate(trainer.learningrate*0.9)
    if prev_cost_adam <= cost_adam:
        trainer_adam.set_learningrate(trainer_adam.learningrate*0.9)

    prev_cost = cost
    prev_cost_adam = cost_adam

print "Done."


