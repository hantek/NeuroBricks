"""
A dataset wrapper should only load its corresponding dataset from a pulically
avaliable format to a user-favourable format. We choose numpy.ndarray as the
ONLY format of output for all datasets. While at the mean time, it should make
sure that the content in the dataset is kept intact. Anything that related to
changing values of the dataset content should be considered as preprocessing
and thus should go into the preprocessing file. For large datasets, a wrapper
should provide an iterator to allow for iterative loading different parts of
the dataset. 

So,
 - here this file is COMPLETELY free from theano, 
 - all wrappers read dataset from various initial formats to numpy.ndarray,
 - they provide an iterator for reading dataset, and
 - they try to reserve as much as possible the raw content of dataset. 

This is the fundamental design we should follow in this file. 


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
import os
import gzip
import cPickle
import numpy
from scipy.misc import imread


class MNIST(object):
    def __init__(self, file_path='/data/lisa/data/mnist.pkl.gz'):
        """
        MNIST is too tiny to have a iterator structure.
        """
        f = gzip.open('/data/lisa/data/mnist.pkl.gz', 'rb')
        self.train_set, self.valid_set, self.test_set = cPickle.load(f)
        f.close()
    
    def get_train_set(self, include_valid=False):
        if include_valid:
            big_train = numpy.concatenate(
                self.train_set[0], self.valid_set[0], axis=0
            )
            big_truth = numpy.concatenate(self.train_set[1], self.valid_set[1])
            return (big_train, big_truth)
        elif:
            return self.train_set
    
    def get_valid_set(self):
        return self.valid_set

    def get_test_set(self):
        return self.test_set

    def get_digit(self, digit=3):
        if digit > 9 or digit < 0:
            raise ValueError("digit has to be an integer in [0, 9].")
        big_train = self.get_train_set(include_valid=True)
        ind = (big_train[1] == digit)
        digit_data = big_train[0][ind]
        digit_truth = big_train[1][ind]
        return (digit_data, digit_truth) 


class CatsnDogs(object):
    def __init__(self, partsize=2500,
                 trainfolderpath="/data/lisa/data/dogs_vs_cats/train",
                 testfolderpath="/data/lisa/data/dogs_vs_cats/test1",
                 validsize=5000,
                 npy_rng=None):
        """
        This is a class with 3 iterators over traim, valid, test sets. The
        validation set is split from training set. All the iterators should do
        the following thing:
        1. read every data sample in a randomly permutated way.
        2. generate a part of data. This allows for subsequent preprocessing.
           Since the images are of various sizes, so during each iteration it
           generates a tuple of numpy arrays, with each array including one
           image. Values in those arrays are completely raw, i.e., uint8 type.
    
        Parameters:
        -------------------
        partsize : int
        Number of images in a generated data part. It is better to ensure that
        this number is dividable for training, validation, and test size.
        
        testfolderpath : str
        Path to unzipped test1 file. contains 12500 files numbered from 1 to
        12500. 

        validsize : int
        To be fair for both cats and dogs, it has to be an even number.
        """
        if not npy_rng:
            npy_rng = numpy.random.RandomState(123)
        assert isinstance(npy_rng, numpy.random.RandomState)
        self.npy_rng = npy_rng
        assert validsize % 2 == 0, "validsize sould be an even number."
        self.validsize = validsize
        self.partsize = partsize
        self.trainfolderpath = trainfolderpath
        self.testfolderpath = testfolderpath

        cat_seq = self.npy_rng.permutation(12500)
        dog_seq = self.npy_rng.permutation(12500)

        self.traincat = cat_seq[:(12500 - validsize / 2)]
        self.validcat = cat_seq[(12500 - validsize / 2):]

        self.traindog = dog_seq[:(12500 - validsize / 2)]
        self.validdog = dog_seq[(12500 - validsize / 2):]

        self.test_seq = numpy.arange(12500)+1
        
        self.reset_generators() 

    def reset_generators(self):
        # pointers indicating which part of iterant to generate.
        self.train_ptr = 0
        self.valid_ptr = 0
        self.test_ptr = 0

    def _read_files(self, folderpath, yield_set):
        current_dir = os.getcwd()
        os.chdir(folderpath)
        img_list = [None, ] * len(yield_set)
        for i in xrange(len(yield_set)):
            img_list[i] = imread(yield_set[i])
        os.chdir(current_dir)
        return tuple(img_list)

    def train_generator(self):
        """A generator for training set."""
        while self.train_ptr < len(self.traincat):
            # if it is the last part, it might be smaller than previous ones.
            yieldsize = min(self.partsize,
                            (len(self.traincat) - self.train_ptr) * 2)
            yield_set = [None for _ in range(yieldsize)]
            yield_truth = numpy.asarray(
                [0, ] * (yieldsize / 2) + [1, ] * (yieldsize / 2))
            
            # generate corresponding file names
            for i in xrange(yieldsize / 2):
                yield_set[i] = 'cat.' + str(self.traincat[self.train_ptr + i]) + '.jpg'
                yield_set[yieldsize / 2 + i] = \
                               'dog.' + str(self.traindog[self.train_ptr + i]) + '.jpg'
            yield_set = numpy.asarray(yield_set)

            # permutation
            permutation_ind = self.npy_rng.permutation(yieldsize)
            yield_set = yield_set[permutation_ind]
            yield_truth = yield_truth[permutation_ind]

            # read the files iteratively
            yield_set = self._read_files(self.trainfolderpath, yield_set)

            # yield
            yield (yield_set, yield_truth)
            self.train_ptr += self.partsize / 2

    def valid_generator(self):
        while self.valid_ptr < len(self.validcat):
            yieldsize = min(self.partsize, (len(self.validcat) - self.valid_ptr) * 2)
            yield_set = [None for _ in range(yieldsize)]
            yield_truth = numpy.asarray(
                [0, ] * (yieldsize / 2) + [1, ] * (yieldsize / 2))

            # generate corresponding file names
            for i in xrange(yieldsize / 2):
                yield_set[i] = 'cat.' + str(self.validcat[self.valid_ptr + i]) + '.jpg'
                yield_set[yieldsize / 2 + i] = \
                               'dog.' + str(self.validdog[self.valid_ptr + i]) + '.jpg'
            yield_set = numpy.asarray(yield_set)

            # permutation
            permutation_ind = self.npy_rng.permutation(yieldsize)
            yield_set = yield_set[permutation_ind]
            yield_truth = yield_truth[permutation_ind]

            # read the files iteratively
            yield_set = self._read_files(self.trainfolderpath, yield_set)

            # yield
            yield (yield_set, yield_truth)
            self.valid_ptr += self.partsize / 2

    def test_generator(self):
        while self.test_ptr < 12500:
            yieldsize = min(self.partsize, (12500 - self.test_ptr))
            yield_set = [None for _ in range(yieldsize)]

            # generate corresponding file names
            for i in xrange(yieldsize):
                yield_set[i] = str(self.test_seq[self.test_ptr + i]) + '.jpg'
            yield_set = numpy.asarray(yield_set)

            # read the files iteratively
            yield_set = self._read_files(self.testfolderpath, yield_set)

            # yield
            yield yield_set
            self.test_ptr += self.partsize


"""
def convert_to_onehot(truth_data):
    """"""
    truth_data is a numpy array.
    """"""
    labels = numpy.unique(truth_data)
    data = numpy.zeros((truth_data.shape[0], len(labels)))
    data[numpy.arange(truth_data.shape[0]), 
         truth_data.reshape(truth_data.shape[0])] = 1
    return data, labels
"""

