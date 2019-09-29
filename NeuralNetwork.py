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
    
    def perform_for_single_input(self, X, y):
        if len(X) != self.number_of_inputs:
            print('input number doesn\'t match')

        #feed forward
        a = [None for _ in range(len(self.number_of_units_at_layer))]

        a[0] = np.array([X]).T

        vfunc = np.vectorize(self.g)
        for i in range(1, len(self.number_of_units_at_layer)):
            a[i-1] = np.append(a[i-1], [[1]], axis=0)
            a[i] = self.weights[i-1].dot(a[i-1])
            a[i] = vfunc(a[i])

        #backprop
        s = [None for _ in range (len(self.number_of_units_at_layer))]
        current_layer = len(self.number_of_units_at_layer) - 1

        s[current_layer] = a[current_layer] - y

        s[current_layer - 1] = self.weights[current_layer-1].T.dot(s[current_layer])
        s[current_layer - 1] = np.multiply(s[current_layer-1], a[current_layer - 1])
        s[current_layer - 1] = np.multiply(s[current_layer-1], 1 - a[current_layer - 1])

        for l in reversed(range(1, len(self.number_of_units_at_layer) - 2)):
            s[l] = self.weights[l].T.dot(s[l+1][:-1,:])
            s[l] = np.multiply(s[l], a[l])
            s[l] = np.multiply(s[l], 1 - a[l])

        self.parray("error", s)

        
        #find gradient
        delta = [None for _ in range(len(self.number_of_units_at_layer))]
        for l in range(len(self.number_of_units_at_layer) - 2):
            delta[l] = np.zeros((self.number_of_units_at_layer[l+1], self.number_of_units_at_layer[l]+1))
            delta[l] += s[l+1][:-1,:].dot(a[l].T)

        delta[current_layer-1] = np.zeros((self.number_of_units_at_layer[current_layer], self.number_of_units_at_layer[current_layer-1]+1))
        delta[current_layer-1] += s[current_layer].dot(a[current_layer-1].T)

        self.parray("grad", delta)



    def show(self):
        for i, x in enumerate(self.weights):
            print('----------------------------------------------')
            print('weight layer ' +str(i) + ' to layer ' + str(i+1) + '. dim = '+ str(x.shape))
            print(x)
            print('----------------------------------------------')
            
    def parray(self, desc, array):
        print('')
        print(desc)
        for i, x in enumerate(array):
            print('----------------------------------------------')
            print(x)
            print('----------------------------------------------')
        print('')


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
