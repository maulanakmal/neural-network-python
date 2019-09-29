import math
import numpy as np

class NeuralNetwork:

    def __init__(self, number_of_inputs):
        self.number_of_units_at_layer = []
        # weight[i] -> weight from layer i to layer i+1
        self.weights = []

        self.number_of_inputs = number_of_inputs
        self.prev_number_of_inputs = number_of_inputs
        self.number_of_units_at_layer.append(number_of_inputs)

    def add_layer(self, number_of_nodes):
        weight = np.random.rand(number_of_nodes, self.prev_number_of_inputs + 1)
        self.weights.append(weight)

        self.prev_number_of_inputs = number_of_nodes
        self.number_of_units_at_layer.append(number_of_nodes)

    # add output layer
    def finish(self):
        weight = np.random.rand(1, self.prev_number_of_inputs + 1)
        self.weights.append(weight)
        self.number_of_units_at_layer.append(1)

    # activation function
    def g(self, x): 
        return 1 / (1 + math.exp(-x))
    
    # derivation of activation function
    def g_der(x):
        return g(x) * (1 - g(x));

    def perform_for_single_input(self, X, y):
        if len(X) != self.number_of_inputs:
            print('input number doesn\'t match')
        #feed forward
        a = [None for _ in range(len(self.number_of_units_at_layer))]

        a[0] = np.array([X.T]).T

        for i in range(1, len(self.number_of_units_at_layer)):
            a[i-1] = np.append(a[i-1], [[1]], axis=0)
            a[i] = self.weights[i-1].dot(a[i-1])

        #backprop
        s = [None for _ in range (len(self.number_of_units_at_layer))]
        current_layer = len(self.number_of_units_at_layer) - 1

        s[current_layer] = a[current_layer] - y
        


    def show(self):
        for i, x in enumerate(self.weights):
            print('----------------------------------------------')
            print('weight layer ' +str(i) + ' to layer ' + str(i+1) + '. dim = '+ str(x.shape))
            print(x)
            print('----------------------------------------------')


def main():
    nn = NeuralNetwork(3)
    nn.add_layer(3)
    nn.add_layer(2)
    nn.finish()

    nn.show()
    
    X = np.array([1,2,3])
    nn.perform_for_single_input(X,1)


if __name__ == "__main__":
    main()
