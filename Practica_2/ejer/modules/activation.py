import numpy as np

class activation(object):
    def __init__(self):
        pass
    def __call__(self, x):
        pass
    def gradient(self,x):
        pass    


class Sigmoid(activation):
    def __call__(self, x):
        exp = 1 + np.exp(-1*x)
        return 1/exp

    def derivate(self, x):
        return np.exp(-x)/((1 + np.exp(-1*x))**2) 


class Linear(activacion):
    def __call__(self, x):
        return x

    def derivate(self, x):
        return np.ones_like(x)


class ReLU(activation):
    def __call__(self, x):
        X = x 
    def gradient(self,x):
        pass 
