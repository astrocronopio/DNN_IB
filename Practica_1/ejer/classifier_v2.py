#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np 

class LinearClassifier(object):
    def __init__(self, eta=0.01, 
                 tolerance=9e-1, 
                 epochs=10, use_bias=False, batch_size=2):

        self.eta=eta
        self.tolerance=tolerance
        self.epochs=epochs
        self.batch_size=batch_size
        self.use_bias=use_bias

    def predict(self, x):
        return np.argmax(self.activacion(x), axis=-1)
    
    def activacion(self,x):
        pass
    def loss_gradient(self, W,x,y):
        pass
    
    def fit(self, x, y):
        n_clasifi=10

        self.im_shape =x.shape[1:]
        # transformo a la imagen a un vector unidimensional
        self.x = np.reshape(x, (x.shape[0], np.prod(self.im_shape)))
        self.x/=255 #Porque las imagenes del cifar10 y mnist varian hasta 255
        
        #Si hay bias, agrego un 1 al final del vector
        if (self.use_bias==True): np.append(self.x, 1) 
        
        #Inicializa pesos
        self.W = np.random.uniform(-0.1,0.15,size=(n_clasifi, self.x.shape[1])) 
        self.y = y

        error_loss=np.zeros(self.epochs)
        error_acc =np.zeros(self.epochs)
        
        for iteracion in range(self.epochs):
            index   =   np.random.randint(0,x.shape[0],size=self.batch_size)
            x_batch =   self.x[index]#self.x[index:final]
            y_batch =   self.y[index]

            L_loss  =   self.bgd(x_batch, y_batch)
            yp      =   self.predict(x_batch)

            error_acc[iteracion] += (yp==y_batch).mean()
            error_loss[iteracion] = L_loss


        return error_loss, 100*error_acc/self.batch_size

    def bgd(self,x_batch, y_batch):
        loss , dW= self.loss_gradient(self.W, x_batch , y_batch)
        self.W+= -self.eta*dW
        return loss

class SVM(LinearClassifier): #Support Vector Machine
    def activacion(self,x):
        super()
        return np.dot(self.W, x.T)+ self.delta

    def loss_gradient(self, W,x,y):
        super()
        self.delta=1
        self.lambda_L2 = 0.005
        
        yp=self.activacion(x)
        y=y.reshape(x.shape[0]) #por si no es como yo quiero

        diff = yp - y[np.newaxis,:] + self.delta
        diff = diff*np.heaviside(diff,0)
        L2= np.mean(self.W*self.W)

        #sumo intra-vector, ahora tengo un [batchsize,(1)]  
        L=diff.sum(axis=-1) 
        loss = np.mean(L) + 0.5*self.lambda_L2*L2

        # 'y' tiene las posiciones de la solucion
        # es genial porque las puedo usar para forzar el 0 donde debe ir
        diff[y, np.arange(x.shape[0])]=0 
        diff[diff>0]=1

        id= np.arange(x.shape[0], dtype=np.int)
        diff[y,id ] -= diff.sum(axis=0)[id]
        print(diff)
        #binary = np.zeros_like(self.W)
        #diff_ = np.zeros_like(self.W)
        dW = np.dot(diff, x)/len(id) + self.lambda_L2*self.W
        print(dW.shape)
        #diff_solucion[y[i]:]+= -np.sum(diff[i])*x[i]
        
        #binary = diff_solucion + diff_
        #binary /=x.shape[0]
        #binary += self.lambda_L2*self.W


        return loss, dW

class SMC(LinearClassifier): #SoftMax Classifier
    def activacion(self,x):
        super()
        return np.dot(x,self.W.T)

    def loss_gradient(self, W,x,y):
        yp=self.activacion(x)
        
        max = np.max(yp)
        yp-+max
        expos= np.exp(yp)
        expo=np.exp(y)
        L = expo/np.sum(expos)

        dW = L*()
        
        
        loss=np.mean(L)
        
        
        
        return loss, dW
