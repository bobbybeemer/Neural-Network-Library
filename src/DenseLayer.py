import numpy

class DenseLayer:
    
    # constructor
    def __init__(self, input_size, output_size):
        # create random weights and random biases
        self.weights = numpy.random.randn(output_size, input_size)
        self.bias = numpy.random.randn(output_size, 1)

    # forward pass
    def forward(self, input):
        # set the layer's input equal to the provided inputs
        self.input = input
        # matrix multiplication
        # the weights multiplied by the inputs, and then add the biases
        return numpy.dot(self.weights, self.input) + self.bias

    #backward propagation
    def backward(self, output_gradient, learning_rate):
        # calculate weight gradient (matrix)
        # matrix multiplication with output gradient and inputs (column vector multiplied by row vector)
        weights_gradient = numpy.dot(output_gradient, self.input.T)
        # create copy of weights matrix so we can use these weights in return
        store_weights = self.weights
        # update the weights to be the current weights - the learning rate times the weights gradient
        self.weights -= learning_rate*weights_gradient
        # update the biases to be the the current biases - the learning rate times the output gradient
        self.bias -= learning_rate*output_gradient
        
        # return the gradient of the inputs so previous layer can do backpropagation
        # gradient of the inputs = weights before being updated transposed times the output gradient
        return numpy.dot(store_weights.T, output_gradient)
