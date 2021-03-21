import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import os.path
import random

class NeuralNetwork(object):
	"""docstring for NeuralNetwork"""
	def __init__(self):
		self.inputLayerSize = 35
		self.outputLayerSize = 4
		self.hiddenLayerSize1 = 20
		self.hiddenLayerSize2 = 10
		self.learning = 0.05
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
		J = (0.5*np.sum((y - self.yHat)**2))
		return J

	def makeWeights(self, W, O):
		WM = np.random.rand(len(O), len(W))
		for x in range(len(O)):
			for y in range(len(W)):
				WM[x][y] = W[y]*O[x]
		return WM

	def Reasignar(self, WS, WP):
		for x in range(WS.shape[0]):
			for y in range(WS.shape[1]):
				WP[x][y] = WP[x][y] - self.learning * WS[x][y]

	def PPN(self, Delta, Pesos):
		"""
		Se multiplican las derivadas para tener en cuenta cuanto influeyen los pesos por cada
		neurona de salida
		"""
		Vector = np.random.rand(Pesos.shape[0])
		for x in range(Pesos.shape[0]):
			Vector[x] = 0
			for y in range(len(Delta)):
				Vector[x] += Delta[y]*Pesos[x][y]

		return Vector

	def BackProp(self, X, y):
		"""
		Primera Capa Oculta
		"""
		self.yHat = self.forward(X)
		dJdO = (- y + self.yHat)
		dOdN = self.yHat * (1 - self.yHat)
		dNdW3 = self.a3
		DeltaO = dJdO*dOdN
		Weights3 = self.makeWeights(DeltaO,dNdW3)
		"""
		Segunda Capa oculta
		"""
		dJtdH = self.PPN(DeltaO,self.W3)
		dHdN = self.a3 * (1 - self.a3)
		dNdW2 = self.a2
		DeltaH2 = dJtdH*dHdN
		Weights2 = self.makeWeights(DeltaH2,dNdW2)
		"""
		Tercera Capa Oculta
		"""
		dNtdO = self.PPN(DeltaH2, self.W2)
		dHidN = self.a2 * (1 * self.a2)
		DeltaH1 = dNtdO * dHidN
		dNdW1 = X
		Weights1 = self.makeWeights(DeltaH1,dNdW1)

		self.Reasignar(Weights3,self.W3)
		self.Reasignar(Weights2,self.W2)
		self.Reasignar(Weights1,self.W1)



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
	
	for y in range(10):
		for x in range(1000):
			n = random.randint(0,9)
			NN.BackProp(X_in[n],Y_out[n])
		print("Epoca: ",y," Loss: ", NN.costFunction(X_in[n],Y_out[n]))
	Y = NN.forward(X_in[0])
	print("Uno: ",Y)
	Y = NN.forward(X_in[1])
	print("Dos: ",Y)
	Y = NN.forward(X_in[2])
	print("Tres: ",Y)
	Y = NN.forward(X_in[3])
	print("Cuatro: ",Y)
	Y = NN.forward(X_in[4])
	print("Cinco: ",Y)
	Y = NN.forward(X_in[5])
	print("Seis: ",Y)
	Y = NN.forward(X_in[6])
	print("Siete: ",Y)
	Y = NN.forward(X_in[7])
	print("Ocho: ",Y)
	Y = NN.forward(X_in[8])
	print("Nueve: ",Y)
	Y = NN.forward(X_in[9])
	print("Cero: ",Y)
	"""
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
	"""