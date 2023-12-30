import math
import Activation

class Sigmoid(Activation):

    def __init__(self):
        sigmoid = lambda x : 1 / (1 + math.exp(-x))
        sigmoid_deriv = lambda x : sigmoid(x) * (1-sigmoid(x))
        super().__init__(sigmoid, sigmoid_deriv)