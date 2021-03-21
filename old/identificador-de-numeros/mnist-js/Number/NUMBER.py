import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import os.path

class NeuralNetwork(object):
	"""docstring for NeuralNetwork"""
	def __init__(self):
		self.inputLayerSize = 35
		self.outputLayerSize = 4
		self.hiddenLayerSize1 = 16
		self.hiddenLayerSize2 = 8

		#weights
		self.W1 = np.random.rand(self.inputLayerSize, self.hiddenLayerSize1)
		self.W2 = np.random.rand(self.hiddenLayerSize1, self.hiddenLayerSize2)
		self.W3 = np.random.rand(self.hiddenLayerSize2, self.outputLayerSize) 

	def forward(self, X):
		#Propagate input
		self.z2 = np.dot(X, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		self.a3 = self.sigmoid(self.z3)
		self.z4 = np.dot(self.a3, self.W3)
		self.yHat = self.sigmoid(self.z4)
		return self.yHat

	def sigmoid(self, z):
		#Apply sigmoid activation function
		return 1/(1+np.exp(-z))

	def sigmoidPrime(self, z):
		#Derivate of sigmoid function
		return self.sigmoid(z)*(1 - self.sigmoid(z))

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
		params = np.concatenate((self.W1.ravel(), self.W2.ravel(), self.W3.ravel()))
		return params

	def setParams(self, params):
		#Set W1 and W2 using a simple parameter vector:
		W1_start = 0
		W1_end = self.hiddenLayerSize1*self.inputLayerSize
		self.W1 = np.reshape(params[W1_start:W1_end],(self.inputLayerSize, self.hiddenLayerSize1))

		W2_end = W1_end + self.hiddenLayerSize1*self.hiddenLayerSize2
		self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize1, self.hiddenLayerSize2))

		W3_end = W2_end + self.hiddenLayerSize2*self.outputLayerSize
		self.W3 = np.reshape(params[W2_end:W3_end], (self.hiddenLayerSize2, self.outputLayerSize))

	def computeGradients(self, X, y):
		dJdW1, dJdW2, dJdW3 = self.costFunctionPrime(X, y)
		dJdW1_vector = dJdW1.ravel()
		dJdW2_vector = dJdW2.ravel()
		dJdW3_vector = dJdW3.ravel()
		Final = np.concatenate((dJdW1_vector, dJdW2_vector, dJdW3_vector))
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
		options = {'maxiter': 1500,"disp" : True}
		_res = optimize.minimize(self.costFunctionWrapper, params0, jac = True, method = "BFGS", args = (X, y), options = options, callback = self.callBack)

		self.NN.setParams(_res.x)
		self.optimizationResuts = _res

if __name__ == "__main__":
	NN = NeuralNetwork()
	X_in = np.array([[0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,1,0],
					 [0,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,0],
					 [1,1,1,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,0,0,0,1,0,1,1,1,0,0],
					 [0,0,0,1,0,0,0,1,1,0,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,0,0,0,1,0,0,0,0,1,0],
					 [0,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,0],
					 [0,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0],
					 [0,0,1,0,0,0,1,1,0,0,1,0,1,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,1,0,0],
					 [0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0],
					 [0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,1,1,1,0],
					 [0,1,1,1,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,0,1,1,1,0]])
	Y_out = np.array([[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[0,0,0,0]])	
	Y = NN.forward(X_in)
	print(Y)
	try:
		print("Lleno")
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
		Y =NN.forward(X_in)
		print(Y)
		for x in range(10):
			Y[x] = Y[x]*x*x 
		print(Y)
		plt.plot(cost)
		plt.grid(1)
		plt.title("Progress of learning of Or Function")
		plt.ylabel("cost")
		plt.xlabel("Iterations")
		plt.show()	
	except:
		print("Vacio")
		Trainner = TrainNN(NN)
		Trainner.train(X_in, Y_out)
		Waves1 = NN.W1.ravel()
		Waves2 = NN.W2.ravel()
		np.savetxt("waves.txt",np.concatenate((Waves1,Waves2)), fmt = "%1.7f")
		np.savetxt("cost.txt", Trainner.J)
		plt.plot(Trainner.J)
		plt.grid(1)
		plt.title("Progress of learning of Or Function")
		plt.xlabel("Iterations")
		plt.ylabel("cost")
		plt.show()
	
	