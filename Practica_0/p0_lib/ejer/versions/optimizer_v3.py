import numpy as np

"""
Versión 3: Esta funciona con el 6 y 7
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
        
        output, reg_sum =self.forwprop(x_batch)

        self.backprop(output, y_batch, x_batch)

        return output, reg_sum

    def update_weights(self, W, gradW):
        super()
        W -= self.lr*gradW 

    def backprop(self, output, y, x): #Back Propagation
        capa2= self.model.capas[-1]
        capa1= self.model.capas[-2]

        ### "Backguard"
        grad2    = self.model.loss_function.gradient(output, y)
        grad_lin = capa2.act.derivate(capa2.S)
        grad2    = grad2*grad_lin

        if capa2.bias:
            xx = np.hstack(((np.ones((len(capa1.S),1) ),capa1.S)))
        else: xx=x

        #Capa 2
        gradw2  = np.dot(grad2.T, xx) + capa2.reg.derivate(capa2.w)
        grad1   = np.dot(grad2, capa2.w)

        #Capa 1

        grad_sig = capa1.act.derivate(xx)                
        grad1 = grad1*grad_sig 

        if capa1.bias:
            grad1 = np.delete(grad1, (0), axis=1)
            xx = np.hstack(((np.ones((len(x),1) ),x)))
        else: xx=x

        gradw1 = np.dot(grad1.T, xx) +  capa1.reg.derivate(capa1.w)


        self.update_weights(capa2.w , gradw2)#w1+= -self.eta*(gradw1)#+  self.lambda_L2*w1
        self.update_weights(capa1.w , gradw1)#w2+= -self.eta*(gradw2)# + self.lambda_L2*w2


    def forwprop(self, X):#Forward Propagation
        reg_sum=0
        S = np.copy(X)
        for capa in self.model.capas:
            S = capa(S)
            reg_sum+= capa.reg(capa.w)
        #print("ultimo scores", S)
        return S, reg_sum