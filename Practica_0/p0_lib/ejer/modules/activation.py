import numpy as np

class activation(object):
    def __init__(self):
        pass
    def __call__(self, x):
        pass
    def gradient(self,x):
        pass    


class sigmoid(activation):
    def __call__(self, x):
        
        exp = 1 + np.exp(-1.0*x)
        return 1/exp

    def derivate(self, x):
        
        return np.exp(-x)*(self.__call__(x)**2)


class Linear(activation):
    def __call__(self, x):
        
        return x

    def derivate(self, x):
        
        return 1.0


class ReLU(activation):
    def __init__(self, delta=0):
        
        self.delta=delta

    def __call__(self, x):
        
        X = np.maximum(x + self.delta, 0)  
        return X

    def derivate(self,x):
        
        X= np.heaviside(x + self.delta, 0)
        return X

class ReLU_Linear(activation):
    def __init__(self, delta=0):
        
        self.delta=delta

    def __call__(self, x):
        
        X = np.maximum(x + self.delta, 0) + x
        return X

    def derivate(self,x):
        
        X= np.heaviside(x + self.delta, 0) + 1
        return X


class Tanh(activation):
    def __call__(self, x):
        
        return  np.tanh(1.0*x)

    def derivate(self, x):
        return 1 - (np.tanh(1.0*x))**2