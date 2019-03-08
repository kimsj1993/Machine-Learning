import numpy as np
import sys
from helper import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def show_images(data):
	"""Show the input images and save them.


	Args:
		data: A stack of two images from traing data with shape (2, 16, 16).
			  Each of the image has the shape (16, 16)

	Returns:
		Do not return any arguments. Save the plots to 'image_1.*' and 'image_2.*' and
		include them in your report
	"""

	### YOUR CODE HERE

	plt.imshow(data[0])
	plt.savefig("image_1")

	plt.imshow(data[1])
	plt.savefig("image_2")
	plt.close()


### END YOUR CODE


def show_features(X, y, save=True):
	"""Plot a 2-D scatter plot in the feature space and save it.

	Args:
		X: An array of shape [n_samples, n_features].
		y: An array of shape [n_samples,]. Only contains 1 or -1.
		save: Boolean. The function will save the figure only if save is True.

	Returns:
		Do not return any arguments. Save the plot to 'train_features.*' and include it
		in your report.
	"""
	### YOUR CODE HERE

	for i in range(len(X)):
		if y[i] == 1:
			plt.scatter(X[i][0], X[i][1], color='r', marker='*')
		else:
			plt.scatter(X[i][0], X[i][1], color='b', marker='+')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.savefig("train_features")


### END YOUR CODE


class Perceptron(object):

	def __init__(self, max_iter):
		self.max_iter = max_iter

	def fit(self, X, y):
		"""Train perceptron model on data (X,y).

		Args:
			X: An array of shape [n_samples, n_features].
			y: An array of shape [n_samples,]. Only contains 1 or -1.

		Returns:
			self: Returns an instance of self.
		"""
		### YOUR CODE HERE
		#since there is already one at first element of X no need to add one
		#self.W = np.zeros(1+X.shape[1])
		self.W = np.zeros(X.shape[1])



		for _ in range (self.max_iter):
			errors = []
			for i in range (len(X)):
				if np.sign(self.W.dot(X[i])) != y[i]:
					errors.append(i)
			random_index = np.random.choice(errors)
			self.W = self.W + y[random_index]*X[random_index]



		### END YOUR CODE

		return self

	def get_params(self):
		"""Get parameters for this perceptron model.

		Returns:
			W: An array of shape [n_features,].
		"""
		if self.W is None:
			print("Run fit first!")
			sys.exit(-1)
		return self.W

	def predict(self, X):
		"""Predict class labels for samples in X.

		Args:
			X: An array of shape [n_samples, n_features].

		Returns:
			preds: An array of shape [n_samples,]. Only contains 1 or -1.
		"""

	### YOUR CODE HERE
		return np.where(np.dot(X, self.W) >= 0.0, 1, -1)
	### END YOUR CODE

	def score(self, X, y):
		"""Returns the mean accuracy on the given test data and labels.

		Args:
			X: An array of shape [n_samples, n_features].
			y: An array of shape [n_samples,]. Only contains 1 or -1.

		Returns:
			score: An float. Mean accuracy of self.predict(X) wrt. y.
		"""
	### YOUR CODE HERE
		p = self.predict(X)
		return np.mean(p == y)
	### END YOUR CODE


def show_result(X, y, W):
	"""Plot the linear model after training.
	   You can call show_features with 'save' being False for convenience.

	Args:
		X: An array of shape [n_samples, 2].
		y: An array of shape [n_samples,]. Only contains 1 or -1.
		W: An array of shape [n_features,].

	Returns:
		Do not return any arguments. Save the plot to 'result.*' and include it
		in your report.
	"""


	### YOUR CODE HERE

	print (W)
	for i in range(len(X)):
		if y[i] == 1:
			plt.scatter(X[i][0], X[i][1], color='r', marker='*')
		else:
			plt.scatter(X[i][0], X[i][1], color='b', marker='+')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()

	#plt.savefig("result")

	### END YOUR CODE


def test_perceptron(max_iter, X_train, y_train, X_test, y_test):
	# train perceptron
	model = Perceptron(max_iter)
	model.fit(X_train, y_train)
	train_acc = model.score(X_train, y_train)
	W = model.get_params()

	# test perceptron model
	test_acc = model.score(X_test, y_test)

	return W, train_acc, test_acc
