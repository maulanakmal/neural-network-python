import math
import numpy as np

class NeuralNetwork:
    number_of_units_at_layer = []
    # weight[i] -> weight from layer i to layer i+1
    weights = []

    def __init__(self, number_of_inputs):
        self.number_of_inputs = number_of_inputs
        self.prev_number_of_output = number_of_inputs
        self.number_of_units_at_layer.append(number_of_inputs)

    def add_layer(self, number_of_nodes):
        weight = np.random.rand(self.prev_number_of_inputs + 1, number_of_nodes)
        self.weights.append(weight)

        self.prev_number_of_inputs = number_of_nodes
        self.number_of_units_at_layer.append(number_of_nodes)

    # add output layer
    def finish():
        self.weights.append(np.random.rand(self.number_of_inputs + 1, 1))

    # activation function
    def g(x): 
        return 1 / (1 + math.exp(-x))
    
    # derivation of activation function
    def g_der(x):
        return g(x) * (1 - g(x));

    def perform_for_single_input(X, y):
        #feed forward
        a = [None for _ in range(len(number_of_units_at_layer))]

        a[0] = X
        for i in range(1, len(number_of_units_at_layer)):
            a[i] = np.zeros((number_of_units_at_layer[i]))
            for unit in range(number_of_units_at_layer[i]):
                for input in range(number_of_units_at_layer[i-1]):
                    a[i][unit] += a[i-1][input] * weights[i][input][unit] + weight[i][-1:][unit]

