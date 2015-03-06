import os
import gzip
import numpy
import theano
import theano.tensor as T
import cPickle
import time

from model import ClassicalAutoencoder
from classifier import LogisticRegression
from train import GraddescentMinibatch
from params import save_params


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is an numpy.ndarray of 2 dimensions (a matrix)
    # witch row's correspond to an example. target is a
    # numpy.ndarray of 1 dimensions (vector)) that have the same length as
    # the number of rows in the input. It should give the target
    # target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(
            numpy.asarray(data_x, dtype=theano.config.floatX),
            borrow=borrow
        )
        shared_y = theano.shared(
            numpy.asarray(data_y, dtype='int64'),
            borrow=borrow
        )
        return shared_x, shared_y

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval









#############
# LOAD DATA #
#############
datasets = load_data('/data/lisa/data/mnist.pkl.gz')

train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x, test_set_y = datasets[2]

npy_rng = numpy.random.RandomState(123)

###############
# BUILD MODEL #
###############

model = ClassicalAutoencoder(
    784, 784, vistype = 'binary', npy_rng = npy_rng
) + ClassicalAutoencoder(
    784, 784, vistype = 'binary', npy_rng = npy_rng
) + ClassicalAutoencoder(
    784, 784, vistype = 'binary', npy_rng = npy_rng
) + LogisticRegression(
    784, 10, npy_rng = npy_rng
)

error_rate = theano.function(
    [], 
    T.mean(T.neq(model.models_stack[-1].predict(), test_set_y)),
    givens = {model.models_stack[0].varin : test_set_x},
)

#############
# PRE-TRAIN #
#############

for i in range(len(model.models_stack)-1):
    print "\n\nPre-training layer %d:" % i
    trainer = GraddescentMinibatch(
        varin=model.varin, data=train_set_x,
        cost=model.models_stack[i].cost(),
        params=model.models_stack[i].params_private,
        supervised=False,
        batchsize=1, learningrate=0.001, momentum=0., rng=npy_rng
    )

    layer_analyse = model.models_stack[i].encoder()
    layer_analyse.draw_weight(patch_shape=(28, 28, 1), npatch=100)
    layer_analyse.hist_weight()
    
    for epoch in xrange(15):
        trainer.step()
        layer_analyse.draw_weight(patch_shape=(28, 28, 1), npatch=100)
        layer_analyse.hist_weight()

save_params(model=model, filename="mnist_sae_784_784_784_10.npy")


#############
# FINE-TUNE #
#############

print "\n\nBegin fine-tune: normal backprop"
bp_trainer = GraddescentMinibatch(
    varin=model.varin, data=train_set_x, 
    truth=model.models_stack[-1].vartruth, truth_data=train_set_y,
    supervised=True, cost=model.models_stack[-1].cost(),
    params=model.params,
    batchsize=1, learningrate=0.1, momentum=0., 
    rng=npy_rng
)
for epoch in xrange(1000):
    bp_trainer.step()
    print "    error rate: %f" % (error_rate())
    pdb.set_trace()

save_params(model=model, filename="mnist_sae_784_784_784_10_ft.npy")
