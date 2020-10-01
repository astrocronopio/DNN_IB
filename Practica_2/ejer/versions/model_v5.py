import numpy as np
"""
Versión 4: NO Funciona con los fors :()
Version 5: Funciona con todos los ejercicios
"""

import modules.activation  as activation 
import modules.layer as layer
import modules.metric as metric
import modules.optimizer as optimizer
import modules.regularizador as regularizador
import modules.loss as loss


class Red(object):
    def __init__(self):
        self.capas=[layer.Entrada("Input")]
        pass

    def predict(self, y):
        pass

    def add(self, layer_neuronas):
        if len(self.capas)>0:
            capa_anterior=self.capas[-1]

            if layer_neuronas.isCon==False:
                capa_anterior.set_ydim(layer_neuronas.get_xdim())
            else:
                capa_anterior.set_ydim(layer_neuronas.get_xdim2())
                
            capa_anterior.ini_weights()
            self.capas.append(layer_neuronas)
        else:
            self.capas.append(layer_neuronas) 
        pass


    def fit(self,   x_train=None, y_train=None, 
                    x_test=None , y_test=None ,
                    batch_size=4,
                    epochs=200  ,
                    opt=optimizer.SGD(lr=0.1),
                    loss_function=loss.MSE(),
                    acc_function =metric.accuracy
                    ):
        
        self.loss_function=loss_function
        self.acc_function=acc_function
        self.opt=opt
        self.batch_size=batch_size

        self.acc_vect = []#np.zeros(epochs)
        self.loss_vect= []#np.zeros(epochs)
        self.pres_vect= []#np.zeros(epochs)
        self.loss_test= []#np.zeros(epochs)

        self.iter_batch= int(x_train.shape[0]/batch_size)

        ultima_capa=self.capas[-1]
        ultima_capa.set_ydim(y_train.shape[1])
        ultima_capa.ini_weights()

        for it in range(epochs):
            
            opt(x_train,y_train,self)

            if np.any(x_test!=None) and np.any(y_test!=None):
                output, reg_sum = self.forwprop(x_test)

                loss = self.loss_function(output, y_test) + reg_sum
                acc  = self.acc_function(output, y_test)
                self.loss_test.append(loss)
                self.pres_vect.append(100*acc)
            

                print("-Epoca {}/{} - loss:{:.4} - loss_test: {:.4} - acc:{:.4} - acc_test:{:.4}".format(
                        it, epochs, 
                        self.loss_vect[-1],
                        self.loss_test[-1],
                        self.acc_vect[-1] ,
                        self.pres_vect[-1]))
            else:    
                print("Epoca {}/{} - loss: {:.4} - acc:{:.4}".format(
                    it, epochs, 
                    self.loss_vect[-1],
                    self.acc_vect[-1] ))


    def backprop(self, output, y, x): #Back Propagation
        capa2= self.capas[-1]
        #capa_media= self.capas[-2]
        capa1= self.capas[1]

        ### "Backguard" ###
        grad    = self.loss_function.gradient(output, y)
        grad_act= capa2.act.derivate(capa2.X)
        grad    = grad*grad_act

        if capa2.bias:
            xx = np.hstack(((np.ones((len(capa2.X),1) ), capa2.X)))
        else: xx=capa2.X


        gradw  = np.dot(grad.T, xx) + capa2.reg.derivate(capa2.w)
        grad   = np.dot(grad, capa2.w)
        capa2.update_weights(self.opt.lr, gradw)

        if capa2.isCon:
            # print(xx.shape)
            # print(xx)
            ind = np.arange(capa2.layer2xdim + 1*capa2.bias, dtype=np.int)
            xx=xx[:,ind]
            grad = grad[:,ind]
            pass

        ####################################
        #print(len(self.capas[-2:1:1]))
        if len(self.capas[-2:1:1])>0:
            for capa_media in self.capas[-2:1:1]:
                grad_act = capa_media.act.derivate(xx)                
                grad = grad*grad_act 
                
                if capa_media.bias:
                    grad = np.delete(grad, (0), axis=1)
                    xx = np.hstack(((np.ones((len(capa_media.X),1) ),capa_media.X)))
                else: xx=capa_media.X
                gradw  = np.dot(grad.T, xx) + capa_media.reg.derivate(capa_media.w)
                grad   = np.dot(grad, capa_media.w)
                capa_media.update_weights(self.opt.lr, gradw)
        
        #Capa 1
        grad_act = capa1.act.derivate(xx)                
        grad = grad*grad_act 

        if capa1.bias:
            grad = np.delete(grad, (0), axis=1)
            xx = np.hstack(((np.ones((len(x),1) ),x)))
        else: xx=x

        gradw = np.dot(grad.T, xx) +  capa1.reg.derivate(capa1.w)
        capa1.update_weights(self.opt.lr, gradw)


    def forwprop(self, X): #Forward Propagation
        reg_sum=0
        S=self.capas[1](X)

        for capa in self.capas[2::1]:
            if capa.isCon:
                S = np.concatenate((S,X), axis=1)
                S = capa(S)
            else:
                S= capa(S)
            reg_sum+= capa.reg(capa.w)

        return S, reg_sum