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
		self.outputLayerSize = 1
		self.hiddenLayerSize = 16

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

def Redondear(X):
	for x in X[0]:
		if x[0] < 0.99:
			x[0] = 0
		else:
			x[0] = 1


if __name__ == "__main__":
	NN0 = NeuralNetwork()
	NN1 = NeuralNetwork()
	NN2 = NeuralNetwork()
	NN3 = NeuralNetwork() 
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
	Y_out3 = np.array([[0],[0],[0],[0],[0],[0],[0],[1],[1],[0]])	
	Y_out2 = np.array([[0],[0],[0],[1],[1],[1],[1],[0],[0],[0]])	
	Y_out1 = np.array([[0],[1],[1],[0],[0],[1],[1],[0],[0],[0]])	
	Y_out0 = np.array([[1],[0],[1],[0],[1],[0],[1],[0],[1],[0]])	

	address = "C:\David\Programacion\Python\ANN\wavesFinished.txt"
	try:
		waves = []
		cost = []
		file = open("waves0Finished.txt", "r")
		filec = open("cost0Finish.txt","r")
		for line in file:
			num = float(line[:-2])
			waves.append(num)
		for value in filec:
			num = value[:-1]
			cost.append(num)
		cost = np.array(cost)
		waves = np.array(waves)
		NN0.setParams(waves)

		waves = []
		cost = []
		file = open("waves1Finished.txt", "r")
		filec = open("cost1Finish.txt","r")
		for line in file:
			num = float(line[:-2])
			waves.append(num)
		for value in filec:
			num = value[:-1]
			cost.append(num)
		cost = np.array(cost)
		waves = np.array(waves)
		NN1.setParams(waves)

		waves = []
		cost = []
		file = open("waves2Finished.txt", "r")
		filec = open("cost2Finish.txt","r")
		for line in file:
			num = float(line[:-2])
			waves.append(num)
		for value in filec:
			num = value[:-1]
			cost.append(num)
		cost = np.array(cost)
		waves = np.array(waves)
		NN2.setParams(waves)

		waves = []
		cost = []
		file = open("waves3Finished.txt", "r")
		filec = open("cost3Finish.txt","r")
		for line in file:
			num = float(line[:-2])
			waves.append(num)
		for value in filec:
			num = value[:-1]
			cost.append(num)
		cost = np.array(cost)
		waves = np.array(waves)
		NN3.setParams(waves)

		Out0 = NN0.forward([X_in])
		Out1 = NN1.forward([X_in])
		Out2 = NN2.forward([X_in])
		Out3 = NN3.forward([X_in])

		Redondear(Out0)
		Redondear(Out1)
		Redondear(Out2)
		Redondear(Out3)

		print(Out0)
		print(Out1)
		print(Out2)
		print(Out3)

		"""
		plt.plot(cost)
		plt.grid(1)
		plt.title("Progress of learning of Or Function")
		plt.ylabel("cost")
		plt.xlabel("Iterations")
		plt.show()	
		"""
	except:
		Trainner0 = TrainNN(NN0)
		Trainner0.train(X_in, Y_out0)
		Trainner1 = TrainNN(NN1)
		Trainner1.train(X_in, Y_out1)
		Trainner2 = TrainNN(NN2)
		Trainner2.train(X_in, Y_out2)
		Trainner3 = TrainNN(NN3)
		Trainner3.train(X_in, Y_out3)
		Waves01 = NN0.W1.ravel()
		Waves02 = NN0.W2.ravel()
		np.savetxt("waves0.txt",np.concatenate((Waves01,Waves02)), fmt = "%1.7f")
		np.savetxt("cost0.txt", Trainner0.J)
		Waves11 = NN1.W1.ravel()
		Waves12 = NN1.W2.ravel()
		np.savetxt("waves1.txt",np.concatenate((Waves11,Waves12)), fmt = "%1.7f")
		np.savetxt("cost1.txt", Trainner1.J)
		Waves21 = NN2.W1.ravel()
		Waves22 = NN2.W2.ravel()
		np.savetxt("waves2.txt",np.concatenate((Waves21,Waves22)), fmt = "%1.7f")
		np.savetxt("cost2.txt", Trainner2.J)
		Waves31 = NN3.W1.ravel()
		Waves32 = NN3.W2.ravel()
		np.savetxt("waves3.txt",np.concatenate((Waves31,Waves32)), fmt = "%1.7f")
		np.savetxt("cost3.txt", Trainner3.J)

		"""
		print(Trainner.J)
		plt.plot(Trainner.J)
		plt.grid(1)
		plt.title("Progress of learning of Or Function")
		plt.xlabel("cost")
		plt.ylabel("Iterations")
		plt.show()
		"""
	