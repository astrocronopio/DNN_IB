import numpy as np

"""
Versión 5: El que rescate del cluster y tenía lindas condiciones iniciales
"""
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
    def __call__(self, x_train, y_train,  model):
        loss, acc=0,0
        for it_ba in range(model.iter_batch):
            index = np.random.randint(0, x_train.shape[0], model.batch_size)
            x_batch =   x_train[it_ba*model.batch_size :(it_ba + 1)*model.batch_size]
            y_batch =   y_train[it_ba*model.batch_size :(it_ba + 1)*model.batch_size]
            
            output, reg_sum = model.forwprop(x_batch)
            
            model.backprop(output, y_batch, x_batch)
            
            loss+= model.loss_function(output, y_batch) + reg_sum
            acc += model.acc_function(output, y_batch)

        model.loss_vect.append(loss/model.iter_batch)
        model.acc_vect.append(100*acc/model.iter_batch)
        #print(output)
 

