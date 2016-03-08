__author__ = 'xiaolin'

# deep belief net

# representation learning using stacked restricted boltzmann machine

import numpy as np
from RBM import RBM
from GaussianBinaryRBM import GRBM
from math import sqrt
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class SRBM(object):

	def __init__(self, input = None, n_ins = 2, hidden_layer_sizes=[3, 3], rng=None, batch_size=100, n_epochs=20, rate=0.05):
		self.x = input
		self.rbm_layers = []
		self.n_layers = len(hidden_layer_sizes) # number of hidden layers

		if rng is None:
			rng = np.random.RandomState(1234)

		assert self.n_layers > 0

		self.layer_record = []
		#construct multi-layer
		for i in xrange(self.n_layers):
			if i == 0:
				# first layer
				input_size = n_ins
				layer_input = self.x
			else:
				input_size = hidden_layer_sizes[i - 1]
				layer_input = self.layer_record[-1]
			# construction of layer_record
			if i == 0:
				print "Now training %d layer" % (i+1)
				ss = sqrt(layer_input.shape[1])

				rbm_layer = GRBM((ss, ss), hidden_layer_sizes[i], rf_shape=(5,5))
				input1 = layer_input.astype(np.float32)
				input1 -= input1.mean(axis=0, keepdims=True)
				input1 /= np.maximum(input1.std(axis=0, keepdims=True), 3e-1)
				batches = input1.reshape(-1, batch_size, input1.shape[1])
				rbm_layer.pretrain(batches, input1, n_epochs=n_epochs, rate=rate)
				self.layer_record.append(rbm_layer.representation)
			else:
				print "Now training %d layer" % (i+1)
				ss = layer_input.shape[1]
				rbm_layer = RBM(ss, hidden_layer_sizes[i])
				input1 = layer_input.astype(np.float32)
				input1 -= input1.mean(axis=0, keepdims=True)
				input1 /= np.maximum(input1.std(axis=0, keepdims=True), 3e-1)
				batches = input1.reshape(-1, batch_size, input1.shape[1])
				rbm_layer.pretrain(batches, input1, n_epochs=n_epochs, rate=rate)
				self.layer_record.append(rbm_layer.representation)
			self.rbm_layers.append(rbm_layer)
def test_dbn():
	datafile = sio.loadmat('data.mat')
	data = datafile['data']
	rng = np.random.RandomState(1234)
	dbn = SRBM(input=data, n_ins = 784, hidden_layer_sizes=[1000], rng=rng)
	tt = data
	test = tt
	test_img = test.astype(np.float32)
	test_img -= test_img.mean(axis=0, keepdims=True)
	test_img /= np.maximum(test_img.std(axis=0, keepdims=True), 3e-1)
	h1 = dbn.rbm_layers[0].encode(test_img)
	#h2 = dbn.rbm_layers[1].encode(h1)
	#r2 = dbn.rbm_layers[1].decode(h2)
	r1 = dbn.rbm_layers[0].decode(h1)
	r1 -= r1.mean(axis=0, keepdims=True)
	r1 /= np.maximum(r1.std(axis=0, keepdims=True), 3e-1)
	a=plt.figure()

	for i in range(10):
		ax = plt.subplot(2, 10, i+1)
		ax.set_xticks([])
		ax.set_yticks([])
		plt.imshow(tt[i+50].reshape(28, 28), cmap=cm.Greys_r, interpolation='none')
		ax=plt.subplot(2,10,i+11)
		ax.set_xticks([])
		ax.set_yticks([])
		plt.imshow(r1[i+50].reshape(28,28), cmap=cm.Greys_r, interpolation='none')



	a.tight_layout()
	plt.show()



if __name__ == '__main__':
	test_dbn()
