import numpy

class Activation:

    # constructor
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # forward pass
    # set the layer's input to the provided input
    # call the activation function on the input
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    # backward propagation
    # element wise multiplication between the output gradient and the derivative of the activation function
    def backward(self, output_gradient, learning_rate):
        return numpy.multiply(output_gradient, self.activation_prime(self.input))