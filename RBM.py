__author__ = 'xiaolin'

# Binary-binary Restricted Boltzmann Machine unit

import numpy as np
import os
import theano
import theano.tensor as T
import theano.sandbox.rng_mrg

os.environ['THEANO_FLAGS'] = 'device=cpu, floatX=float32'
class RBM(object):
	def __init__(self, input=None, n_vis=2, n_hid=3, W=None, v_bias=None, h_bias=None, rng=None):

		self.dtype = theano.config.floatX
		self.n_vis = n_vis
		self.n_hid = n_hid
		if rng is None:
			rng = np.random.RandomState(123)
		if W is None:
			Wmag = 4 * np.sqrt(6. / (self.n_vis + self.n_hid))
			W = rng.uniform(
				low=-Wmag, high=Wmag, size=(self.n_vis, self.n_hid)
			).astype(self.dtype)
		self.theano_rng = theano.sandbox.rng_mrg.MRG_RandomStreams(1234)

		if h_bias is None:
			h_bias = np.zeros(self.n_hid, dtype=self.dtype)

		if v_bias is None:
			v_bias = np.zeros(self.n_vis, dtype=self.dtype)


		W = W.astyoe(self.dtype)
		h_bias = h_bias.astype(self.dtype)
		v_bias = v_bias.astype(self.dtype)

		self.W = theano.shared(W, name='W')
		self.h_bias = theano.shared(h_bias, name='h_bias')
		self.v_bias = theano.shared(v_bias, name='v_bias')

		self.Winc = theano.shared(np.zeros_like(W), name='Winc')
		self.h_bias_inc = theano.shared(np.zeros_like(h_bias), name='h_bias_inc')
		self.v_bias_inc = theano.shared(np.zeros_like(v_bias), name='v_bias_inc')

	def propup(self, vis):
		pre_act = T.dot(vis, self.W) + self.h_bias
		return pre_act, T.nnet.sigmoid(pre_act)

	def propdown(self, hid):
		pre_act = T.dot(hid, self.W.T) + self.v_bias
		return pre_act, T.nnet.sigmoid(pre_act)

	def sampHgivenV(self, vis):
		_, hidprob = self.propup(vis)
		hidsamp = self.theano_rng.binomial(
			size=hidprob.shape, n=1, p=hidprob, dtype=self.dtype
		) #hidden unit sampling
		return hidprob, hidsamp


	# define RBM updates
	def get_updates(self, data, lr = 0.1, weightcost= 2e-4, momentum=0.5):

		numcases = T.cast(data.shape[0], self.dtype)
		lr = T.cast(lr, self.dtype)
		weightcost = T.cast(weightcost, self.dtype)
		momentum = T.cast(momentum, self.dtype)

		# positive phase
		poshidprob, poshidsamp = self.sampHgivenV(data)

		posw = T.dot(data.T, poshidprob) / numcases
		pos_vis = T.mean(data, axis=0)
		pos_hid = T.mean(poshidprob, axis=0)

		# negative phase
		_, negdata = self.propdown(poshidsamp)
		_, neghidprob = self.propup(negdata)
		negw = T.dot(negdata.T, neghidprob)
		neg_vis = T.mean(negdata, axis=0)
		neg_hid = T.mean(neghidprob, axis=0)

		cross_entropy = 



