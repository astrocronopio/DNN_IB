import numpy  as np

""" Esta versión 3"""
import modules.activation  as activation
import modules.metric as metric
import modules.model as model
import modules.optimizer as optimizer
import modules.regularizador as regularizador


class BaseLayer():
    def __init__(self, name="No name"):
        self.name=name

    def __call__(self, x):
        pass
    
    def __str__(self):
        return self.name


class Layer(BaseLayer):
    def update_weights(self): 
        pass



class Dense(Layer):
    def __init__(self, 
                 neuronas   = 1, 
                 act        = 1,
                 reg        = None,
                 name       = "No Name",
                 bias       = False):
        super()
        
        self.neuronas   = neuronas
        self.act        = act 
        self.reg        = reg   
        self.name       = name  
        self.bias       = bias
        self.output_size= 0

    def ini_weights(self):
        self.w= np.random.uniform(-1.0,1.0,size=(self.output_size, self.neuronas + 1*self.bias))

    def __call__(self, x):
        super()
        self.S = self.act(self.dot(self.w, x)) 
        return self.S

    def dot(self, W, x):
        xx = np.hstack(((np.ones((len(x),1) ),x))) if self.bias else x
        return np.dot(xx, W.T)

def Concatenate(BaseLayer):
    def __call__(self):
        pass

    def get_xdimextra(self):
        pass
