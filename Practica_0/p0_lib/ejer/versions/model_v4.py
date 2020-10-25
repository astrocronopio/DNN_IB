import numpy as np
"""
Versión 4: NO Funciona con los fors :()
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
        if len(self.capas)>1:
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
                loss= self.loss_function(output, y_test) + reg_sum
                acc = self.acc_function(output, y_test)
                self.loss_test.append(loss)
                self.pres_vect.append(100*acc)
            

                print("Epoca {}/{} - loss:{:.4} - loss_test: {:.4} - acc:{:.4} - acc_test:{:.4}".format(
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
        capa_final= self.capas[-1]
        capa1= self.capas[-2]
        capa0= self.capas[-3]

        ### "Backguard"
        grad        = self.loss_function.gradient(output, y)
        grad_lin    = capa_final.act.derivate(capa_final.S)
        grad        = grad*grad_lin

        if capa_final.bias:
            xx = np.hstack(((np.ones((len(capa1.S),1) ),capa1.S)))
        else: 
            xx=capa1.S

        #Capa 2
        gradw2  = np.dot(grad.T, xx) + capa_final.reg.derivate(capa_final.w)
        self.opt.update_weights(capa_final.w , gradw2)

        #Capa 1
        grad   = np.dot(grad, capa_final.w)
        capa_anterior=capa1

        for capa in self.capas[-2:1:-1]: #Desde el ultimo hasta el primero
            grad_sig = capa_anterior.act.derivate(xx)                
            grad = grad*grad_sig 

            if capa_anterior.bias:
                grad = np.delete(grad, (0), axis=1)
                xx = np.hstack(((np.ones((len(capa.S),1) ),capa.S)))
            else: xx=capa.S

            if capa_anterior.isCon:
                pass
                            
            gradw1 = np.dot(grad.T, xx) +  capa_anterior.reg.derivate(capa_anterior.w)
            self.opt.update_weights(capa_anterior.w , gradw1)
            capa_anterior=capa

    def forwprop(self, X):#Forward Propagation
        reg_sum=0
        S = X
        for capa in self.capas[1::1]:
            if capa.isCon:
                 S[:np.arange(capa.get_xdim2())] = capa(S)
                 S[:np.arange(capa.get_xdim2(),capa.get_xdim1)] = X
            else:
                S= capa(S)
            reg_sum+= capa.reg(capa.w)
        return S, reg_sum