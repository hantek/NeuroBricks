"""
This file provides a general solutions for passing numerical values between mod-
els and saving/loading parameters with files.

TODO:
Consider to use pkl and tar.gz to store param list of numpy.ndarrays.
Deal with params_private problem.
"""
import numpy
import theano

def set_params(model, newparams):
    def inplaceupdate(x, new):
        x[...] = new
        return x

    paramscounter = 0
    for p in model.params:
        pshape = p.get_value().shape
        pnum = numpy.prod(pshape)
        p.set_value(inplaceupdate(p.get_value(borrow=True),
            newparams[paramscounter:paramscounter+pnum].reshape(*pshape)),
            borrow=True)
        paramscounter += pnum

def get_params(model):
    return numpy.concatenate([p.get_value().flatten() for p in model.params])

def save_params(model, filename):
    numpy.save(filename, get_params(model))

def load_params(model, filename):
    set_params(model, numpy.load(filename))
