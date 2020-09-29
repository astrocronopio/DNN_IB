import numpy as np
"""
Versión 3 antes de concat
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
            capa_anterior.output_size = layer_neuronas.neuronas
            capa_anterior.ini_weights()
            self.capas.append(layer_neuronas)
        else:
            self.capas.append(layer_neuronas) 
        pass


    def fit(self,   x_train=None, y_train=None, 
                    x_test=None , y_test=None ,
                    batch_size=4,
                    epochs=200  ,
                    opt=optimizer.SGD(lr=0.005),
                    loss_function=loss.MSE(),
                    acc_function =metric.accuracy):
        
        self.loss_function=loss_function
        self.acc_function=acc_function

        self.acc_vect = []#np.zeros(epochs)
        self.loss_vect= []#np.zeros(epochs)
        self.pres_vect= []#np.zeros(epochs)
        self.loss_test= []#np.zeros(epochs)

        self.iter_batch= int(x_train.shape[0]/batch_size)

        ultima_capa=self.capas[-1]
        ultima_capa.output_size = y_train.shape[1]
        ultima_capa.ini_weights()

        for it in range(epochs):
            loss, acc=0,0
            for it_ba in range(self.iter_batch):
                x_batch =   x_train[it_ba :(it_ba + 1)*batch_size]
                y_batch =   y_train[it_ba :(it_ba + 1)*batch_size]

                output, reg_sum = opt(x_batch, y_batch, self)
                        
                loss+= self.loss_function(output, y_batch) + reg_sum
                acc += self.acc_function(output, y_batch)

            self.loss_vect.append(loss/self.iter_batch)
            self.acc_vect.append(100*acc/self.iter_batch)
                

            if np.any(x_test!=None) and np.any(y_test!=None):
                output, reg_sum = opt.forwprop(x_test)
                loss= self.loss_function(output, y_test) + reg_sum
                acc = self.acc_function(output, y_test)
                self.loss_test.append(loss)
                self.pres_vect.append(100*acc)
            

                print("Epoca {}/{} - loss:{} - loss_test: {} - acc:{} - acc_test:{}".format(
                        it, epochs, 
                        self.loss_vect[-1],
                        self.loss_test[-1],
                        self.acc_vect[-1] ,
                        self.pres_vect[-1]))
            else:    
                print("Epoca {}/{} - loss: {} - acc:{}".format(
                    it, epochs, 
                    self.loss_vect[-1],
                    self.acc_vect[-1] ))

