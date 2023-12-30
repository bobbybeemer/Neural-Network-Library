import Activation
import numpy

class Tanh(Activation):
    
    # constructor
    def __init__(self):
        tanh = lambda x : numpy.tanh(x)
        tanh_prime = lambda x : 1 - numpy.tanh(x)**2
        super().__init__(tanh, tanh_prime)
    

    
