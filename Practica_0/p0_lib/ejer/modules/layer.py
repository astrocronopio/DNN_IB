import numpy  as np

""" Esta versión 3"""
import modules.activation  as activation
import modules.metric as metric
import modules.model as model
import modules.optimizer as optimizer
import modules.regularizador as regularizador

import time as time
import copy as copy

class BaseLayer():
    def __init__(self, name="No name"):
        self.name=name
        self.S=[]

    def __call__(self, x):
        pass
    
    def __str__(self):
        return self.name


class Layer(BaseLayer):
    def __init__(self, 
                neuronas   = 1, 
                act        = 1,
                reg        = regularizador.L2(0.0),
                name       = "No Name",
                bias       = False,
                isCon      = False):
        self.neuronas   = neuronas
        self.act        = act 
        self.reg        = reg   
        self.name       = name  
        self.bias       = bias
        self.isCon      = isCon


    def get_ydim(self):
        return self.output_size

    def set_ydim(self,out):
        self.output_size=out

    def get_xdim(self):
        return self.neuronas

    def set_xdim(self,n):
        self.neuronas=n

    def update_weights(self, lr, gradW): 
        self.w -= lr*gradW 


class Entrada(Layer):
    def __call__(self, x):
        self.S=[1]
        self.w=1
        return self.S

    def ini_weights(self):
        pass


class Dense(Layer):
    def ini_weights(self):
        self.w= np.random.normal(0,2,size=(self.output_size, self.neuronas + 1*self.bias))

    def __call__(self, x):
        self.X=x
        XX = self.dot(self.w, x)
        YY = self.act(XX) 
        return YY

    def dot(self, W, x):
        if self.bias:
            xx = np.hstack(((np.ones((len(x),1) ),x))) 
        else: xx=x

        return np.dot(xx, W.T)


class ConcatInput(Dense):
    def __init__(self, input_size, Layer2):
        self.neuronas   = input_size + Layer2.get_xdim()
        self.layer2xdim = Layer2.get_xdim()
        self.layer1xdim = input_size
        self.act        = Layer2.act 
        self.reg        = Layer2.reg   
        self.name       = Layer2.name  
        self.bias       = Layer2.bias
        self.isCon      = True

    def get_xdim1(self):
        return self.layer1xdim
    
    def get_xdim2(self):
        return self.layer2xdim
    
