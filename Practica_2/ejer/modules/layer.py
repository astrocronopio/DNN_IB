import numpy  as np

import modules.activation  as activation 
import modules.metric as metric
import modules.model as model
import modules.optimizer as optimizer
import modules.regularizador as regularizador


class BaseLayer():
    def __init__():
        pass
    
    def get_ydim(self):
        pass

    def set_ydim(self):
        pass

    def get_xdim(self):
        pass

    def set_xdim(self):
        pass
    

class Entrada(BaseLayer):
    def __init__():
        pass
    
    def get_ydim(self):
        pass
    
    def set_ydim(self):
        pass



class Layer(BaseLayer):
    def __init__(self, neuronas=0, 
                 activation= None, 
                 input_size= None,
                 reg       = None):
        super()
        pass

    def get_weights(self):
        pass
    
    def update_weights(self): 
        pass


class Dense(Layer):
    def ini_weights():
        pass

    def __call__():
        pass

    def prod():
        pass



def Concatenate(BaseLayer):
    def __call__(self):
        pass

    def get_xdimextra(self):
        pass
