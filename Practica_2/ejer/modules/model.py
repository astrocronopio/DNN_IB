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

    def add(self, layer_neuronas):
        if len(self.capas)>0:
            capa_anterior=self.capas[-1]
            capa_anterior.output_size = layer_neuronas.neuronas
            capa_anterior.ini_weights()
            self.capas.append(layer_neuronas)
        else:
            self.capas.append(layer_neuronas) 
        pass


    def fit(self,   x_train=None, y_train=None, 
                    x_test=None , y_test=None ,
                    batch_size=1,
                    epochs=200  ,
                    opt=optimizer.SGD(lr=0.05),
                    loss_function=loss.MSE(),
                    acc_function =metric.accuracy):
        
        self.loss_function=loss_function
        self.acc_function=acc_function

        self.acc_vect = []#np.zeros(epochs)
        self.loss_vect= []#np.zeros(epochs)
        self.pres_vect= []#np.zeros(epochs)
        self.loss_test= []#np.zeros(epochs)

        iter_batch= int(x_train.shape[0]/batch_size)

        capa_anterior=self.capas[-1]
        capa_anterior.output_size = y_train.shape[1]
        capa_anterior.ini_weights()

        for it in range(epochs):
            for _ in range(iter_batch):
                index   =   np.random.randint(0, x_train.shape[0], batch_size)
                x_batch =   x_train[index]
                y_batch =   y_train[index]

                opt(x_batch, y_batch, self)

            if x_test!=None and y_test!=None:
                loss= self.loss_function(output, y_batch)
                acc = self.acc_function(output, y_batch)
                self.loss_vect.append(loss)
                self.acc_vect.append(100*acc)

            print("Epoca {}/{} - loss: {} - acc:{}".format(
                    it, epochs, 
                    self.loss_vect[-1],
                    self.acc_vect[-1] ))
                


