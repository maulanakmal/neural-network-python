# neural-network-python

This project is an implementation of neural network. Currently only implementing mini batch gradient descent. Contributions are welcomed.

## Installation

This project uses numpy. just grab it using pip :).

```bash
pip install numpy
```

## Usage

```python
# Constructor requires the number of unit in input layer and the learning rate
nn = NeuralNetwork(3, 0.01)

# Add a layer of 3 units
nn.add_layer(3)

# Add another layers
nn.add_layer(3)
nn.add_layer(2)

# Let it know that you have done
nn.finish()

# Run the mini batchgradient descent 
nn.mini_batch_gradient_descent(X, Y, batch_size, number_of_epochs)

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
