"""
You deside how the data set should be pre-processed, and how large the subset to
be used. So, here this file is COMPLETELY free from theano. It reads dataset
from its various initial format to numpy.ndarray.

for data:
A 2-D vector (x) with each samples flattened into ROW. So x is of size
(num_sample, num_dim)

A view of the initial data size should be company with x.

for truth:
A 1-D vector (y_labels) with numbers starting from 0 and continuously to total number of
class.

A dictionary with key=class name and value=class number should be
company with y_labels.


for truth (regression):
TODO
"""

import numpy

def convert_to_onehot(truth_data):
    """
    truth_data is a numpy array.
    """
    labels = numpy.unique(truth_data)
    data = numpy.zeros((truth_data.shape[0], len(labels)))
    data[numpy.arange(truth_data.shape[0]), 
         truth_data.reshape(truth_data.shape[0])] = 1
    return data, labels

def load_mnist(file_path='/data/lisa/data/mnist.pkl.gz'):
    import gzip
    import cPickle
    f = gzip.open('/data/lisa/data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set
