""" 
    ejer3.py
    Versión 4: Antes de agregar la clase loss y activation
    Versión 5: Separé las activaciones y losses en otro modulo
               También agregué la activación de la segunda capa.
    Versión 6: Separé la clase classifier al modulo classifier.py    
"""

"""
Classifier.py
    Vesión 1: Cree modulo
"""

import numpy as np
from keras import datasets
np.random.seed(54)
import matplotlib.pyplot as plt
import modules.activation as act
import modules.loss as los


            
def flattening(x, y, n_clasifi, max_train ):
    X= np.copy(x) 
    X= np.reshape(X, (X.shape[0], np.prod(x.shape[1:])))
    X= (X - max_train)/255
    
    X= np.hstack(( (np.ones((len(X),1) )), X)) 
    
    Y = np.zeros(shape=(y.shape[0], n_clasifi))
    y_aux  = np.copy(y).reshape(y.shape[0])

    Y[np.arange(Y.shape[0]), y_aux]=1
    return X,Y


######

class Classifier(object):
    def __init__(self,  eta         =   0.01, 
                        tolerance   =   9e-1, 
                        epochs      =   10,
                        use_bias    =   False,
                        batch_size  =   2,
                        lambda_L2   =   0.5):

        self.eta        =   eta
        self.tolerance  =   tolerance
        self.epochs     =   epochs
        self.batch_size =   batch_size
        self.use_bias   =   use_bias
        self.lambda_L2  =   lambda_L2

    def predict(self, scores):
        ajax= np.argmax(scores, axis=1)
        return ajax#output

    def accuracy(self, y_pred, y_true):
        acc = (y_pred.T==y_true.T)
        return np.mean(acc)
    
    def loss_function(self, scores, y_true):
        pass

    def fit(self, x, y, x_test, y_test, 
            act_function, loss_function,
            act_function2):

        self.acc_vect = np.zeros(self.epochs)
        self.loss_vect= np.zeros(self.epochs)
        self.pres_vect= np.zeros(self.epochs)
        self.loss_test= np.zeros(self.epochs)

        iter_batch= int(x.shape[0]/self.batch_size)

        w1 = np.random.uniform(-0.001, 0.001, size=(100,(x.shape[1])))
        w2 = np.random.uniform(-0.001, 0.001, size=((10, (w1.shape[0]+1))))


        for it in range(self.epochs): 
            loss,acc=0,0         
            for it_ba in range(iter_batch):
                
                index   =   np.random.randint(0, x.shape[0], self.batch_size)
                x_batch =   x[index]
                y_batch =   y[index]
                
                ####Forward####
                S1= act_function(np.dot(x_batch, w1.T))                
                S1= np.hstack(((np.ones((len(S1),1) ),S1))) 

                S2= act_function2(np.dot(S1,w2.T))
                """Hasta acá bien"""
                ####regularization##

                reg1= np.sum(w1*w1)
                reg2= np.sum(w2*w2)
                reg = reg1+reg2

                #loss and acc
                
                loss += loss_function(S2,y_batch) + 0.5*self.lambda_L2*reg

                S2_out = self.predict(S2)
                y_batch_out = self.predict(y_batch)

                acc  += self.accuracy(S2_out, y_batch_out)
                
                ###"Backguard"

                grad2 = loss_function.gradient(S2, y_batch)
                grad_lin=act_function2.derivate(S2)
                grad2=grad2*grad_lin

                #Capa 2

                gradw2  = np.dot(grad2.T, S1) + self.lambda_L2*w2
                grad1    = np.dot(grad2, w2)

                #Capa 1
                grad_sig = act_function.derivate(S1)                
                grad1 = grad1*grad_sig 
                grad1 = np.delete(grad1, (0), axis=1)
                gradw1 = np.dot(grad1.T, x_batch) +  self.lambda_L2*w1


                w1+= -self.eta*(gradw1)#+  self.lambda_L2*w1
                w2+= -self.eta*(gradw2)# + self.lambda_L2*w2

            self.loss_vect[it]=loss/iter_batch
            self.acc_vect[it]=100*acc/iter_batch

            S1_test= act_function(np.dot(x_test,w1.T))

            S1_test= np.hstack(((np.ones((len(S1_test),1) ),S1_test)))
            S2_test= np.dot(S1_test, w2.T)

            S2_tout= self.predict(S2_test)
            y_test_out=self.predict(y_test)

            self.pres_vect[it] = 100*self.accuracy(S2_tout, y_test_out)
            self.loss_test[it] = loss_function(S2_test,y_test) + 0.5*self.lambda_L2*reg

            print("Epoch: {}/{} - acc_test:{:.4} - loss:{:.4} - acc:{:.4}".format(it, 
            self.epochs, 
            self.pres_vect[it],
            self.loss_vect[it], 
            self.acc_vect[it]))
            
