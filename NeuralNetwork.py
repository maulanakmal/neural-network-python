import math
import random
import numpy as np

class NeuralNetwork:
    weights = []
    number_of_layer = 0
    units = []
    def __init__(self, number_of_inputs):
        self.number_of_inputs = number_of_inputs
        self.weights.append(np.random.rand(1,1)) # dummy
        self.units.append(number_of_inputs)

    def add_layer(self, number_of_nodes):
        self.weights.append(np.random.rand(self.number_of_inputs, number_of_nodes + 1))
        self.number_of_inputs = number_of_nodes
        self.units.append(number_of_nodes)
        self.number_of_layer += 1

    def finish():
        self.weights.append(np.random.rand(self.number_of_inputs, 1))

    # activation function
    def g(x): 
        return 1 / (1 + math.exp(-x))
    
    # derivation of activation function
    def g_der(x):
        return g(x) * (1 - g(x));

    def perform_for_single_input(X):
        layers = [] #output of each layer
        layers.append(X) # X = vector of number_of_inputs input

        # feed forward
        for i in range(1, self.number_of_layer + 1):
            layers.append([0 for _ in range(units[i])])
            for j in range(units[i]):
                for k in range(len(a[i-1])):
                    layers[i][j] += a[i-1][k] * weights[i][k][j]

        # back prob

