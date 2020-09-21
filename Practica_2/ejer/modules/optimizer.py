import numpy as np

"""
Versión 4
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
        self.model.capas[0](x_batch)
        
        output, reg_sum =self.forwprop(x_batch)

        self.backprop(output, y_batch)

        return output, reg_sum

    def update_weights(self, W, gradW):
        super()
        W -= self.lr*gradW 

    def backprop(self, output, y): #Back Propagation
        # flip_capas=self.model.capas[-1::-1]:
        # capa_anterior=self.model.capas[-1]
        # grad    = self.model.loss_function.gradient(output, y)
        # grad_act = capa_anterior.act.derivate(capa_anterior.S)
        # grad = grad*grad_act
        
        # for i in range(len(flip_capas)):
        #     if capa_anterior.bias and i>0:
        #         xx = np.hstack(((np.ones((len(flip_capas[i].S),1) ),flip_capas[i].S)))
        #     else: xx=x  

        #     if i>0:
        #         gradw  = np.dot(grad.T, xx) + capa_anterior.reg.derivate(capa_anterior.w)
        #         self.update_weights(capa_anterior.w , gradw)
        #         capa_anterior=flip_capas[i]           

        #     if i>0: grad   = np.dot(grad, capa_anterior.w)
        #     else:  grad    = self.model.loss_function.gradient(output, y) 
            
        #     grad_act = flip_capas[i].act.derivate(flip_capas[i].S)
        #     grad = grad*grad_act

            
        #     if flip_capas[i].bias:
        #         grad = np.delete(grad, (0), axis=1)


       
        #primera capa
        #if i==(len(flip_capas)-1):
        #                xx = np.hstack(((np.ones((len(x),1) ),x)))

        capa2= self.model.capas[-1]
        capa1= self.model.capas[-2]
        capa0= self.model.capas[-3]

        ### "Backguard"
        grad    = self.model.loss_function.gradient(output, y)
        grad_lin = capa2.act.derivate(capa2.S)
        grad    = grad*grad_lin

        if capa2.bias:
            xx = np.hstack(((np.ones((len(capa1.S),1) ),capa1.S)))
        else: xx=x

        #Capa 2
        gradw2  = np.dot(grad.T, xx) + capa2.reg.derivate(capa2.w)
        self.update_weights(capa2.w , gradw2)

        #Capa 1
        grad   = np.dot(grad, capa2.w)
        grad_sig = capa1.act.derivate(xx)                
        grad = grad*grad_sig 

        if capa1.bias:
            grad = np.delete(grad, (0), axis=1)
            xx = np.hstack(((np.ones((len(capa0.S),1) ),capa0.S)))
        else: xx=capa0.S

        gradw1 = np.dot(grad.T, xx) +  capa1.reg.derivate(capa1.w)
        self.update_weights(capa1.w , gradw1)


    def forwprop(self, X):#Forward Propagation
        reg_sum=0
        S = np.copy(X)
        for capa in self.model.capas[1::1]:
            S = capa(S)
            reg_sum+= capa.reg(capa.w)
        #print("ultimo scores", S)
        return S, reg_sum