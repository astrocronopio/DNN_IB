import numpy as np

"""
V1 No funciona
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
    def __call__(self, x_batch, y_batch,  model):
        self.model=model
        
        output=self.forwprop(x_batch)
        
        loss= self.model.loss_function(output, y_batch)
        acc = self.model.acc_function(output, y_batch)

        self.model.loss_vect.append(loss/y_batch.shape[0])
        self.model.acc_vect.append(100*acc/y_batch.shape[0])
        
        self.backprop(output, y_batch, x_batch)

        pass

    def update_weights(self, W, gradW):
        super()
        return W - self.lr*gradW 

    def backprop(self, output, y, x): #Back Propagation
        capa2= self.model.capas[-1]
        capa1= self.model.capas[-2]
        
        grad2 = self.model.loss_function.gradient(output, y)
        #grad_act = capa.act.derivate(capa.S)
        #grad2 = grad_act*grad2 # 4x1
        W= capa.w #1x3
        
        #Capa 2
        capa= self.model.capas[-2]
        if capa.bias:
            xx = np.hstack(((np.ones((len(capa.S),1) ),capa.S)))
        else: xx=x
        
        gradw2  = np.dot(grad2.T, xx)
        self.update_weights(W,gradw2)

        grad1 = np.dot(grad2, W)
        ##

        #Capa 1
        W=capa.w
        grad_act = capa.act.derivate(capa.S)   

        if capa.bias:
            grad1 = np.delete(grad1, (0), axis=1)
            xx = np.hstack(((np.ones((len(x),1) ),x)))
        else: xx=x
        grad1 = grad1*grad_act 
        # print("W", W.shape)
        # exit()
        
        gradw1 = np.dot(grad1.T, xx) + capa.reg.derivate(W) 

        self.update_weights(W,gradw1)

        # capa= self.model.capas[-1]
        # grad= self.model.loss_function.gradient(output, y)

        # grad_act=capa.act.derivate(capa.S)
        # grad = grad*grad_act 
        # W=capa.w

        # print(len(self.model.capas[-2::-1]))
        # for capa in self.model.capas[-2::-1]:
        #     dw = np.dot(grad.T, capa.S)
        #     self.update_weights(W,dw)
            
        #     grad_act = capa.act.derivate(capa.S)
        #     grad=grad*grad_act      
            
        #     if capa.bias:
        #         grad= np.delete(grad, (0), axis=1)
        #     W=capa.w


    def forwprop(self, X):#Forward Propagation
        S = np.copy(X)
        for capa in self.model.capas:
            S = capa(S)
        #print("ultimo scores", S)
        return S