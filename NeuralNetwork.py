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
        weight = np.random.rand(self.prev_number_of_inputs + 1, number_of_nodes)
        self.weights.append(weight)

        self.prev_number_of_inputs = number_of_nodes
        self.number_of_units_at_layer.append(number_of_nodes)

    # add output layer
    def finish(self):
        self.weights.append(np.random.rand(self.number_of_inputs + 1, 1))
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

        a[0] = X
        for i in range(1, len(self.number_of_units_at_layer)):
            a[i] = np.zeros((self.number_of_units_at_layer[i]))
            for unit in range(self.number_of_units_at_layer[i]):
                for input in range(self.number_of_units_at_layer[i-1]):
                    a[i][unit] += a[i-1][input] * self.weights[i-1][input][unit] + self.weights[i-1][self.number_of_units_at_layer[i-1]][unit]
                    a[i][unit] = self.g(a[i][unit])


    def show(self):
        print(self.number_of_units_at_layer)
        for i, x in enumerate(self.weights):
            print('---------------------------')
            print('weight layer ' +str(i) + ' to layer ' + str(i+1))
            print(x)
            print('---------------------------')


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
