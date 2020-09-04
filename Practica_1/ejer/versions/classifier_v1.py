#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np 

class LinearClassifier(object):
    def __init__(self, eta=0.01, tolerance=9e-1, 
                      epochs=10, use_bias=False, batch_size=2):
        self.eta=eta
        self.tolerance=tolerance
        self.epochs=epochs
        self.batch_size=batch_size
        self.use_bias=use_bias

    def predict(self, x):
        return np.argmax(self.activacion(x))
    
    def activacion(self,x):
        pass
    def loss_gradient(self, W,x,y):
        pass
    
    def fit(self, x, y):
        n_clasifi=10

        self.im_shape =x.shape[1:]
        # transformo a la imagen a un vector unidimensional
        self.x = np.reshape(x, (x.shape[0], np.prod(self.im_shape)))/255
        
        #Si hay bias, agrego un 1 al final del vector
        if (self.use_bias==True): np.append(self.x, 1) 
        
        #Inicializa pesos
        self.W = np.random.uniform(-0.01,0.015,size=(n_clasifi, self.x.shape[1])) 
        self.y = y

        error_loss=np.zeros(self.epochs)
        error_acc =np.zeros(self.epochs)
        
        for iteracion in range(self.epochs):
            index   =   np.random.randint(0,x.shape[0],size=self.batch_size)
            #final   =   index+self.batch_size
            x_batch =   self.x[index]#self.x[index:final]
            y_batch =   self.y[index]

            L_loss  =   self.bgd(x_batch, y_batch)
            yp      =   self.predict(x_batch)

            print(yp==y_batch)
            error_acc[iteracion] += (yp==y_batch).mean()
            error_loss[iteracion] = L_loss
            print(L_loss)

        return error_loss, 100*error_acc/self.batch_size

    def bgd(self,x_batch, y_batch):
        loss , dW= self.loss_gradient(self.W, x_batch , y_batch)
        self.W+= -self.eta*dW
        return loss

class SVM(LinearClassifier): #Support Vector Machine
    def activacion(self,x):
        super()
        return np.dot(x, self.W.T)+ self.delta

    def loss_gradient(self, W,x,y):
        super()
        self.delta=-1
        self.lambda_L2 = 0.005
        yp=self.activacion(x)

        #print(yp)

        diff = yp - y[:,np.newaxis] + self.delta
        diff = diff*np.heaviside(diff,0)
        L2= np.mean(self.W*self.W)
        
        # 'y' tiene las posiciones de la solucion
        # es genial porque las puedo usar para forzar el 0 donde debe ir
        diff[np.arange(x.shape[0]), y]=0 
        
        #sumo intra-vector, ahora tengo un [batchsize,(1)]  
        L=diff.sum(axis=-1) 
        loss=np.mean(L) + 0.5*self.lambda_L2*L2

        diff_solucion = np.zeros_like(self.W)
        diff_ = np.zeros_like(self.W)

        for i in range(x.shape[0]):
            diff_ += np.dot(diff[i][:,np.newaxis], np.transpose(x[i])[np.newaxis,:])
            aux=np.sum(diff[i])
            diff_solucion[y[i]:]+= -aux*x[i]
        
        binary = diff_solucion + diff_
        binary /=x.shape[0]
        binary += self.lambda_L2*self.W

        dW = binary

        return loss, dW

class SMC(LinearClassifier): #SoftMax Classifier
    def activacion(self,x):
        return np.dot(x,self.W.T)

    def loss_gradient(self, W,x,y):
        yp=self.activacion(x)
        loss=np.mean(L)
        return loss, dW
