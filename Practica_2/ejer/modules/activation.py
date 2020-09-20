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


class Linear(activation):
    def __call__(self, x):
        return x

    def derivate(self, x):
        return np.ones_like(x)


class ReLU(activation):
    def __init__(self, delta=1):
        super()
        self.delta=delta

    def __call__(self, x):
        X = np.max(x + self.delta, 0)  
        return X

    def derivate(self,x):
        X= np.heaviside(x + self.delta, 0)
        return X

class Tanh(activation):
    def __call__(self, x):
        self.X = np.tanh(x)
        return self.X

    def derivate(self, x):
        return 1 - self.X**2