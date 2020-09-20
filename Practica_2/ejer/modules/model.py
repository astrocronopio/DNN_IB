import numpy as np

import modules.activation  as activation 
import modules.layer as layer
import modules.metric as metric
import modules.optimizer as optimizer
import modules.regularizador as regularizador
import modules.loss as loss


class Red(object):
    def __init__(self):
        self.capas=[]
        pass

    def predict(self, y):

        pass

    def add(self, layer):
        self.capas.append(layer)
        pass


    def fit(self,   x_train,    y_train, 
                    x_test=None,y_test=None ,
                    batch_size=4,
                    epochs=200,
                    opt=optimizer.SGD(lr=0.05),
                    loss=loss.MSE()):

        for it in range(epochs):
            pass
        pass

    def backprop(self): #Back Propagation
        pass
    
    def forwprop(self): #Forward Propagation
        pass

        