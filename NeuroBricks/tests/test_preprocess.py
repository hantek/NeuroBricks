import numpy
import pickle
import theano
from preprocess import PCA
from preprocess import SubtractMean
from preprocess import SubtractMeanAndNormalize


def test_PCA():
    # first, a 2-d data, testing the reconstruction and whitening
    random_state = 123
    rng = numpy.random.RandomState(random_state)
    data1 = rng.multivariate_normal(
        mean=[0, 0], cov=[[15.0, .0], [.0, 0.1]], size=1000
    )
    data2 = numpy.dot(
        data1, 
        numpy.asarray(
            [[numpy.cos(numpy.pi/9), -numpy.sin(numpy.pi/9)], 
             [numpy.sin(numpy.pi/9),  numpy.cos(numpy.pi/9)]]
        )
    )
    data = numpy.vstack((data1, data2))
    
    pca_obj = PCA()
    pca_obj.fit(data, whiten=False)
    pcaed_data = pca_obj.forward(data)
    recons_data = pca_obj.backward(pcaed_data)
    assert numpy.allclose(data, recons_data)

    pca_obj.fit(data, whiten=True)
    pcaed_data = pca_obj.forward(data)
    recons_data = pca_obj.backward(pcaed_data)
    assert numpy.allclose(data, recons_data)
    assert numpy.allclose(pcaed_data.std(0), numpy.asarray((1., 1.)))

    
    # Non-centered
    data_nc = data + numpy.asarray((10, 5))
    pca_obj = PCA()
    pca_obj.fit(data_nc, whiten=False)
    pcaed_data = pca_obj.forward(data_nc)
    recons_data = pca_obj.backward(pcaed_data)
    assert numpy.allclose(data_nc, recons_data)

    pca_obj.fit(data_nc, whiten=True)
    pcaed_data = pca_obj.forward(data_nc)
    recons_data = pca_obj.backward(pcaed_data)
    assert numpy.allclose(data_nc, recons_data)
    assert numpy.allclose(pcaed_data.std(0), numpy.asarray((1., 1.)))

    # second, real data, testing numerical stability
    data = pickle.load(open('/data/lisa/data/mnist/mnist.pkl', 'r'))
    data = data[0][0]
    pca_obj = PCA()
    pca_obj.fit(data, whiten=False)
    recons_data = pca_obj.backward(pca_obj.forward(data))
    assert numpy.allclose(data, recons_data)
    pca_obj.fit(data, whiten=True)
    pcaed_data = pca_obj.forward(data)
    recons_data = pca_obj.backward(pcaed_data)
    assert numpy.allclose(pcaed_data.std(0)[:700].mean(), 1.0)
    assert numpy.allclose(data, recons_data)

    pca_obj = PCA()
    pca_obj.fit(data, whiten=False, retain=500)
    pcaed_data = pca_obj.forward(data)
    assert pcaed_data.shape == (50000, 500)
    pca_obj.fit(data, whiten=True, retain=500)
    pcaed_data = pca_obj.forward(data)
    assert numpy.allclose(pcaed_data.std(0), numpy.ones(500))
    assert pcaed_data.shape == (50000, 500)

    pca_obj = PCA()
    pca_obj.fit(data, whiten=False, retain=0.99)
    recons_data = pca_obj.backward(pca_obj.forward(data))
    # calculation of std ratio is still not very accurate. 
    assert (numpy.sum(recons_data.std(0)) - \
            numpy.sum(pca_obj.std_fracs[pca_obj.retain-1] * data.std(0)))**2 \
           < 1e-2 * numpy.sum(data.std(0))
    pca_obj.fit(data, whiten=True, retain=0.99)
    pcaed_data = pca_obj.forward(data)
    assert numpy.allclose(pcaed_data.std(0), numpy.ones(pcaed_data.shape[1]))


def test_SubtractMean():
    data_small = numpy.arange(15).reshape(5, 3).astype(theano.config.floatX)
    data_big = numpy.arange(150).reshape(50, 3).astype(theano.config.floatX)
    
    process_layer = SubtractMean(n_in=3)
    small = process_layer.output().eval({process_layer.varin:data_small})
    big = process_layer.output().eval({process_layer.varin:data_big})

    assert numpy.sum((small.mean(axis=0) - big.mean(axis=0))**2) == 0
    assert numpy.sum(small.mean(axis=1) ** 2) == 0
    assert numpy.sum(big.mean(axis=1) ** 2) == 0


def test_SubtractMeanAndNormalize():
    data_small = numpy.arange(18).reshape(6, 3).astype(theano.config.floatX)
    data_small[:3, :] *= 10
    
    process_layer = SubtractMeanAndNormalize(n_in=3)
    small = process_layer.output().eval({process_layer.varin: data_small})
    assert numpy.sum(small.mean(axis=1) ** 2) == 0
    assert numpy.allclose(numpy.sum(small.std(axis=1) ** 2), 6)
