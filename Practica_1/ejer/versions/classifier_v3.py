#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np 
#np.random.seed(40)

import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})

class LinearClassifier(object):
    def __init__(self, eta=0.01, 
                 tolerance=9e-1, 
                 epochs=10, use_bias=False,
                 batch_size=2, lambda_L2 = 0.5):

        self.eta=eta
        self.tolerance=tolerance
        self.epochs=epochs
        self.batch_size=batch_size
        self.use_bias=use_bias
        self.lambda_L2 = 0.05

    def predict(self, x):
        return np.argmax(self.activacion(x), axis=0)
    
    def activacion(self,x):
        pass
    def loss_gradient(self,x,y):
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
        self.W = np.random.uniform(-1,1,size=(n_clasifi, self.x.shape[1])) 
        self.y = y

        self.error_loss=np.zeros(self.epochs)
        self.error_acc =np.zeros(self.epochs)
        
        for it in range(self.epochs):
            index   =   np.random.randint(0,x.shape[0],size=self.batch_size)
            x_batch =   self.x[index]#: index + self.batch_size]#self.x[index:final]
            y_batch =   self.y[index]#: index + self.batch_size]
            L_loss  =   self.bgd(x_batch, y_batch)
            self.error_acc[it] += 100*np.mean((self.predict(x_batch) == y_batch))
            self.error_loss[it] = L_loss
        #return self.error_loss, 100*self.error_acc

    def bgd(self,x_batch, y_batch):
        loss , dW = self.loss_gradient(x_batch , y_batch)
        self.W+= -self.eta*dW
        return loss


class SVM(LinearClassifier): #Support Vector Machine
    def activacion(self,x):
        super()
        return np.dot(self.W, x.T)+ self.delta

    def loss_gradient(self,x,y):
        super()
        self.delta=1#-1
        L2= np.sum(self.W*self.W)

        id= np.arange(x.shape[0], dtype=np.int)
        yp=self.activacion(x)
        y=y.reshape(x.shape[0]) #por sino es como yo quiero

        diff = yp - yp[y,id] + self.delta
        diff = np.maximum(diff, 0)
        diff[y, np.arange(x.shape[0])]=0 

        #sumo intra-vector, ahora tengo un [batchsize,(1)]  
        L=diff.sum(axis=0)
        loss = np.mean(L) + 0.5*self.lambda_L2*L2

        # 'y' tiene las posiciones de la solucion 
        # es genial porque las puedo usar para forzar el 0 donde debe ir
        diff=np.heaviside(diff,0)
        diff[y, id] -= diff.sum(axis=0)[id]

        dW = np.dot(diff, x)/x.shape[0] + self.lambda_L2*self.W
        return loss, dW

class SMC(LinearClassifier): #SoftMax Classifier
    def activacion(self,x):
        super()
        return np.dot(self.W, x.T)

    def SoftMax(self, y, yp):
        max_val =   np.max(yp,axis=0)
        yp-= max_val
        y-=max_val
        ind= np.arange(y.shape[0], dtype=np.int)
        expos   =   np.exp(yp)
        #print(expos.shape)
        expo    =   np.exp(y)
        #print(expos.shape)

        sf = expo[ind]/np.sum(expos[ind]) 
        print(sf.shape)
        return sf        

    def loss_gradient(self,x,y):
        super()
        y=y.reshape(x.shape[0]) 
        ind= np.arange(x.shape[0], dtype=np.int)
        L2= np.sum(self.W*self.W)
        yp= self.activacion(x)
        
        diff = 0.0*yp[:]

        sf= self.SoftMax(yp[y,ind], yp)

        L = np.log(sf)
        
        X = np.dot(np.diag(L), x)

        #dW = np.zeros_like(self.W)
        diff[y,ind] = -1 

        #X= np.sum(x, axis=0)
        #print(x.shape)
        #print(L.shape)  
        #diff = np.dot(L,x) 
        #print(diff.shape)
        #dW += pepe#np.dot(dW, X.T) 
        dW = np.dot(diff, X) + self.lambda_L2*self.W 
        loss=np.mean(L)  + 0.5*self.lambda_L2*L2
        
        return loss, dW
