__author__ = 'xiaolin'
import theano.sandbox.rng_mrg
import theano.tensor as T
from theano import function
theano_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(1233)
a = theano_rng.normal(size=(1, 2), avg=(-1, 4), std=1)
b = function([],a)
print b()