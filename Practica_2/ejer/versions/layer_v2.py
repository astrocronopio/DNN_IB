import numpy  as np
""" Esta versión 2 tiene las propuestas de Lucca"""
import modules.activation  as activation
import modules.metric as metric
import modules.model as model
import modules.optimizer as optimizer
import modules.regularizador as regularizador

import copy

class BaseLayer():
    def __init__(self, name="No name"):
        self.name=name

    def __call__(self, x):
        pass
    
    def __str__(self):
        return self.name

    def get_ydim(self):
        pass

    def set_ydim(self):
        pass

    def get_xdim(self):
        pass

    def set_xdim(self):
        pass
    

class Entrada(BaseLayer):
    
    def get_ydim(self):
        pass
    
    def set_ydim(self):
        pass



class Layer(BaseLayer):

    def get_weights(self):
        pass
    
    def update_weights(self): 
        pass



class Dense(Layer):
    def __init__(self, 
                 neuronas   = 1, 
                 act        = 1, 
                 input_size = 1,
                 reg        = None,
                 name       = "No Name",
                 bias       = False):
        super()
        
        self.neuronas   = neuronas
        self.act        = act 
        self.input_size = input_size
        self.reg        = reg   
        self.name       = name  
        self.bias       = bias
        #self.ini_weights()
        self.output_size=0
        #self.w=[]   

    def ini_weights(self):
        if self.bias:
            self.w= np.random.uniform(-0.1,0.1,size=(self.input_size + 1 , self.output_size))
        else:
            self.w= np.random.uniform(-0.1,0.1,size=(self.input_size, self.output_size))
        pass

    def __call__(self, x):
        super()
        W = self.w
        self.S = self.act(self.dot(W, x)) 
        return self.S

    def dot(self, W, x):
        if self.bias:
            xx = np.hstack(((np.ones((len(x),1) ),x)))
            return np.dot(xx, W)
        else:
            return np.dot(x, W)

        pass


def Concatenate(BaseLayer):
    def __call__(self):
        pass

    def get_xdimextra(self):
        pass
