import numpy as np


import modules.activation  as activation 
import modules.layer as layer
import modules.metric as metric
import modules.model as model
import modules.regularizador as regularizador



class optimizer(object):
    def __init__(self, lr):
        self.lr = lr
    
    def __call__(self, X,Y, model):
        pass

    def update_weights(self, W, gradW):
        pass

class SGD(optimizer):
    def __call__(self, X,Y,model):
        #model.backward(X,Y)
        pass

    def update_weights(self, W, gradW):
        super()
        return W + self.lr*gradW