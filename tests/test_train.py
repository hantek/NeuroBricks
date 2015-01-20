import os
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import cPickle

from layer import LinearLayer
from classifier import LogisticRegression
from model import ClassicalAutoencoder
from train import Dropout


def test_Dropout():
    npy_rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(123)
    data_x = theano.shared(
        100 * npy_rng.normal(0, 1, [1000, 50]).astype(theano.config.floatX))
    data_y = theano.shared(
        npy_rng.randint(0, 10, 1000))
   

    ae = ClassicalAutoencoder(
        50, 70, vistype='real', hidtype='binary', tie=True
    )
    sl = LinearLayer(50, 70) + LogisticRegression(70, 10)
    # sl.print_layer()
    lg = LogisticRegression(50, 10)
    # lg.print_layer()

    ae_recon = theano.function(
        [],
        ae.reconstruction(),
        givens={ae.varin: data_x}
    )
    sl_output = theano.function(
        [],
        sl.output(),
        givens={sl.varin: data_x}
    )
    lg_output = theano.function(
        [],
        lg.output(),
        givens={lg.varin: data_x}
    )
    recon_before_dropout = ae_recon()
    output_before_dropout = sl_output()
    lgoutput_before_dropout = lg_output()

    dropout_ae = Dropout(ae, [0.2, 0.5], theano_rng=theano_rng)
    dropout_sl = Dropout(sl, [0.7, 0.5], theano_rng=theano_rng)
    dropout_lg = Dropout(lg, [0.5], theano_rng=theano_rng)
    # dropout_ae.dropout_model.print_layer()
    # dropout_sl.dropout_model.print_layer()
    # dropout_lg.dropout_model.print_layer()

    ae_recon = theano.function(
        [],
        ae.reconstruction(),
        givens={ae.varin: data_x}
    )
    sl_output = theano.function(
        [],
        sl.output(),
        givens={sl.varin: data_x}
    )
    lg_output = theano.function(
        [],
        lg.output(),
        givens={lg.varin: data_x}
    )
    recon_after_dropout = ae_recon()
    output_after_dropout = sl_output()
    lgoutput_after_dropout = lg_output()

    assert numpy.allclose(recon_before_dropout, recon_after_dropout)
    assert numpy.allclose(output_before_dropout, output_after_dropout)
    assert numpy.allclose(lgoutput_before_dropout, lgoutput_after_dropout)
