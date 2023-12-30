import numpy

def mse(y_true, y_predict):
    return numpy.mean(numpy.power(y_true-y_predict, 2))

def mse_derivative(y_true, y_predict):
    return 2 / numpy.size(y_true) * (y_predict-y_true)