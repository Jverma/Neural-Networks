# -*- coding: utf-8 -*-
#	Neural Networks
#	Author - Janu Verma
#	jv367@cornell.edu
#	@januverma


from __future__ import division
import numpy as np 
import sklearn.datasets as datasets
from sklearn import cross_validation
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score




class NeuralNetwork:
	"""
	Implements an artificial neural network. 
	"""
	def __init__(self, numHiddenLayers=1, nHiddenUnits=25, maxIter=500):
		self.numHiddenLayers = numHiddenLayers
		self.numHiddenUnits = nHiddenUnits
		self.maxIter = maxIter
		self.layers = [self.numHiddenUnits for x in range(numHiddenLayers)]

	def activation(self, z):
		"""
		The activation function. 
		The default choice is sigmoid function.
		"""	
		return 1.0 / (1.0 + np.exp(-z))


	def costFunction(self, Y, Y_pred):
		"""
		Compute the cost function.
		"""	
		n = Y.shape[0]
		J = (-1.0/n)*(Y * np.log(Y_pred) + ((1-Y) * np.log(1 - Y_pred))).sum()
		return J

	def forwardFeed(self, X, weightList):
		"""
		Compute the predicted values. 
		"""	
		m = X.shape[0]

		# Activation functions of all the layers
		activationPrediction = []

		# Input activation
		activationPrediction.append(X)
	
		# Loop through weights at layer
		for i, weight in enumerate(weightList):
			# Add bias unit
			if (activationPrediction[i].ndim == 1):
				activationPrediction.resize(1, activationPrediction[i].shape[0])
			ones = np.ones((activationPrediction[i].shape[0], 1))
			activationPrediction[i] = np.append(ones, activationPrediction[i], 1)

			# compute output of the layer
			z = activationPrediction[i].dot(weight.T)
			activationPred = self.activation(z)
			activationPrediction.append(activationPred)

		return activationPrediction	


	def backProp(self, predictedValues, trainingValues, weightList, lmbda = 0):
		"""
		Compute the weights of each layer adjusted to the error of the output. 
		"""	
		l = len(weightList)
		predictedOutput = predictedValues[l]
		numObservations = len(predictedOutput)

		errorList = []

		# Get error for the output layer
		delta = predictedOutput - trainingValues
		if (delta.ndim == 1):
			delta.resize(1,delta.shape[0])
		errorList.append(delta)
		
		# Errors of the hidden layers 
		for i in range(l-1, 0, -1):
			layerWeight = weightList[i]
			delta = delta.dot(layerWeight[:,1:])
			delta = delta * (predictedValues[i][:,1:] * (1 - predictedValues[i][:,1:]))
			errorList.append(delta)

		# reverse the error list so that errorList[i] is the 
		# errror (delta) that weightList[i] causes on activationPrediction[i+1]
		errorList.reverse()	


		# calculate the gradient using errors and activations
		# i indexes the weight from layer i to layer i+1
		weightGradient = []
		for i in range(l):
			alpha = errorList[i].T
			weightGradient.append(alpha.dot(predictedValues[i]))

		# modify weights for regularization
		# bais is not regularized
		regWeight = [np.zeros_like(weight) for weight in weightList]
		for i,weight in enumerate(weightList):
			regWeight[i][:,1:] = weight[:,1:]

		# add regularization penalty
		for i in range(l):
			weightGradient[i] = weightGradient[i]*(1.0/numObservations) + lmbda * regWeight[i]

		return weightGradient		


	def fit(self, X_train, Y_train, learningRate=0.01, X_test=None, Y_test=None):
		"""
		fits the training data to the neural network. 
		Calls the predict and backProp methods repeatedly. 
		It tracks error and modify prediction accordingly.
		"""	
		numInputUnits = X_train.shape[1]
		numOutputUnits = Y_train.shape[1]

		# List of units for each layer. 
		unitCounts = [numInputUnits]
		unitCounts.extend(self.layers)
		unitCounts.append(numOutputUnits)

		# initialize the weights
		weightList = [2 * (np.random.rand(unit_count, unitCounts[l-1] + 1) - 0.5) for l,unit_count in enumerate(unitCounts) if l > 0]
		l = len(weightList)

		learningRates = []
		learningRates.append(learningRate)

		# Initial weight change terms
		weightChange = [np.zeros_like(x) for x in weightList]


		# List of cost functions
		CostFunctionList = [0]*self.maxIter
		CostFunctionTestList = [0]*self.maxIter

		# Initial forward propagation
		activationTrain = self.forwardFeed(X_train, weightList)
		Y_predicted = activationTrain[l]
	
		# Initial cost
		CostFunctionList[0] = self.costFunction(Y_train, Y_predicted)


		# no error for input layer
		for i in range(1, self.maxIter):
			weightGradient = self.backProp(activationTrain, Y_train, weightList)
			for j,weightGrad in enumerate(weightGradient):
				weightChange[j] = learningRate * weightGrad 
				weightList[j] = weightList[j] - weightChange[j]

			# Update
			activationTrain = self.forwardFeed(X_train, weightList)
			Y_predictedNew = activationTrain[l]


			# Check if the cost decreased
			costNew = self.costFunction(Y_train, Y_predictedNew)

			CostFunctionList[i] = costNew
			
		return weightList	



	def predict(self, X_train, Y_train, X_test, learning_rate=0.5):
		weightList = self.fit(X_train, Y_train, learning_rate)
		results = self.forwardFeed(X_test, weightList)
		l = len(weightList)
		outputLayer = results[l]
		predictions = []
		for i in range(X_test.shape[0]):
			predictions.append(np.argmax(outputLayer[i]))	
		return np.array(predictions)
			









#################################################################################################
#################################################################################################
# TESTING ON IRIS AND IMAGES DATASETS

def check_iris():
	iris = datasets.load_iris()
	X = iris.data
	Y = iris.target

	X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.4)

	lb = LabelBinarizer()
	Y_train = lb.fit_transform(Y_train)

	out = NeuralNetwork()
	dot = out.predict(X_train, Y_train, X_test)
	print accuracy_score(Y_test, dot)


def check_digits():
	digits = datasets.load_digits()
	X = digits.data
	Y = digits.target
	X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.4)

	lb = LabelBinarizer()
	Y_train = lb.fit_transform(Y_train)

	out = NeuralNetwork(maxIter=500)
	dot = out.predict(X_train, Y_train, X_test)
	print accuracy_score(Y_test, dot)

	
check_digits()	
#check_iris()




















