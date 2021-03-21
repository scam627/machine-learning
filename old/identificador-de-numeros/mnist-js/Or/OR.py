import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import os.path

class NeuralNetwork(object):
	"""docstring for NeuralNetwork"""
	def __init__(self):
		self.inputLayerSize = 2
		self.outputLayerSize = 1
		self.hiddenLayerSize = 3

		#weights
		self.W1 = np.random.rand(self.inputLayerSize, self.hiddenLayerSize)
		self.W2 = np.random.rand(self.hiddenLayerSize, self.outputLayerSize) 

	def forward(self, X):
		#Propagate input
		self.z2 = np.dot(X, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		self.yHat = self.sigmoid(self.z3)
		return self.yHat

	def sigmoid(self, z):
		#Apply sigmoid activation function
		return 1/(1+np.exp(-z))

	def sigmoidPrime(self, z):
		#Derivate of sigmoid function
		return np.exp(-z)/((1 + np.exp(-z))**2)

	def costFunction(self, X, y):
		self.yHat = self.forward(X)
		J = (0.5*sum((y - self.yHat)**2))
		return J

	def costFunctionPrime(self, X, y):
		self.yHat = self.forward(X)
		delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
		dJdW2  = np.dot(self.a2.T, delta3)

		delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
		dJdW1  = np.dot(X.T, delta2) 

		return dJdW1, dJdW2



	# getParams, setParams, computeGradients es solo para comprobar la derivada esta correcta
	def getParams(self):
		params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
		return params

	def setParams(self, params):
		#Set W1 and W2 using a simple parameter vector:
		W1_start = 0
		W1_end = self.hiddenLayerSize*self.inputLayerSize
		self.W1 = np.reshape(params[W1_start:W1_end],(self.inputLayerSize, self.hiddenLayerSize))

		W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
		self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

	def computeGradients(self, X, y):
		dJdW1, dJdW2 = self.costFunctionPrime(X, y)
		dJdW1_vector = dJdW1.ravel()
		dJdW2_vector = dJdW2.ravel()
		Final = np.concatenate((dJdW1_vector, dJdW2_vector))
		return Final

class TrainNN(object):
	"""docstring for TrainNN"""
	def __init__(self, NN):
		self.NN = NN

	def costFunctionWrapper(self, params, X, y):
		self.NN.setParams(params)
		cost = self.NN.costFunction(X, y)
		grad = self.NN.computeGradients(X, y)
		return cost, grad

	def callBack(self, params):

		self.NN.setParams(params)
		#Add local costs to the cost of the nn
		self.J.append(self.NN.costFunction(self.X, self.y))

	def train(self, X, y):
		self.X = X
		self.y = y
		#creamos un vector con todos los pesos
		params0 = self.NN.getParams()
		self.J = []
		options = {'maxiter': 200,"disp" : True}
		_res = optimize.minimize(self.costFunctionWrapper, params0, jac = True, method = "BFGS", args = (X, y), options = options, callback = self.callBack)

		self.NN.setParams(_res.x)
		self.optimizationResuts = _res

if __name__ == "__main__":
	NN = NeuralNetwork()
	X_in = np.array([[0,0],[0,1],[1,0],[1,1]])
	Y_out = np.array([[0],[1],[1],[1]])	
	address = "C:\David\Programacion\Python\ANN\wavesFinished.txt"
	print(NN.forward(X_in))
	try:
		waves = []
		cost = []
		file = open("wavesFinished.txt", "r")
		filec = open("costFinish.txt","r")
		for line in file:
			num = float(line[:-2])
			waves.append(num)
		for value in filec:
			num = value[:-1]
			cost.append(num)
		cost = np.array(cost)
		waves = np.array(waves)
		NN.setParams(waves)
		print(NN.forward([X_in]))
		plt.plot(cost)
		plt.grid(1)
		plt.title("Progress of learning of Or Function")
		plt.ylabel("cost")
		plt.xlabel("Iterations")
		plt.show()	
	except:
		Trainner = TrainNN(NN)
		Trainner.train(X_in, Y_out)
		Waves1 = NN.W1.ravel()
		Waves2 = NN.W2.ravel()
		np.savetxt("waves.txt",np.concatenate((Waves1,Waves2)), fmt = "%1.7f")
		np.savetxt("cost.txt", Trainner.J)
		print(Trainner.J)
		plt.plot(Trainner.J)
		plt.grid(1)
		plt.title("Progress of learning of Or Function")
		plt.xlabel("cost")
		plt.ylabel("Iterations")
		plt.show()

	