#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np 
np.random.seed(42)

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
                 epochs=10,
                 batch_size=2, lambda_L2 = 0.5):

        self.eta=eta
        self.epochs=epochs
        self.batch_size=batch_size
        #self.use_bias=use_bias
        self.lambda_L2 = lambda_L2

    def predict(self, x):
        # X = np.copy(x) 
        # X = np.reshape(X, (X.shape[0], np.prod(X.shape[1:])))
        return np.argmax(self.activacion(X), axis=0)

    def activacion(self,x):
        pass
    def loss_gradient(self,x,y):
        pass
    
    def fit(self, x, y, x_test, y_test):
        n_clasifi=10

        # #Si hay bias, agrego un 1 al final del vector
        # if (self.use_bias==True):
        #     X= np.hstack(( (np.ones((len(x),1) )), X)) 
        #     X_test=np.hstack(( (np.ones((len(x_test),1) )), X_test))  #np.append(X_test, 1)
        
        #Inicializa pesos
        self.W = np.random.uniform(-0.1,0.1,size=(n_clasifi, X.shape[1])) 
        Y = np.copy(y)
        self.y_test = np.copy(y_test)
        self.error_loss=np.zeros(self.epochs)
        self.error_acc =np.zeros(self.epochs)
        self.error_pres=np.zeros(self.epochs)
        
        iter_batch= int(X.shape[0]/self.batch_size)

        for it in range(self.epochs):
            print("Epoca: ",it)
            for it_ba in range(iter_batch): 
                index   =   np.random.randint(0, x.shape[0], self.batch_size)
                x_batch =   X[index]
                y_batch =   Y[index]

                self.error_loss[it]  +=   self.bgd(x_batch, y_batch)
                self.error_acc[it]  += 100.0*np.mean((self.predict(x_batch) == y_batch))
            
            self.error_pres[it] = 100.0*np.mean((self.predict(X_test) == y_test))

        self.error_acc/=iter_batch 
        self.error_loss/=iter_batch

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
        diff[y, id]=0 

        # 'y' tiene las posiciones de la solucion 
        # es genial porque las puedo usar para forzar el 0 donde debe ir

        #sumo intra-vector, ahora tengo un [batchsize,(1)]  
        L=diff.sum(axis=0)
        loss = np.mean(L) + 0.5*self.lambda_L2*L2

        diff=np.heaviside(diff,0)
        diff[y, id] -= diff.sum(axis=0)[id]

        dW = np.dot(diff, x)/x.shape[0] + self.lambda_L2*self.W
        return loss, dW

class SMC(LinearClassifier): #SoftMax Classifier
    def activacion(self,x):
        super()
        return np.dot(self.W, x.T)

    def loss_gradient(self,x,y):
        super()
        L2= np.sum(self.W*self.W)
        #y=y.reshape(x.shape[0]) 
        
        yp = self.activacion(x)
        yp-= np.max(yp,axis=0)
        
        ind= np.arange(self.batch_size, dtype=np.int)

        expo    =   np.exp(yp)
        expo_sum=   np.sum(expo, axis=0)
        
        diff    =   expo/expo_sum
        diff[y,ind] += -1

        L = -yp[y,ind] + np.log(expo_sum[ind])

        dW = np.dot(diff, x)/self.batch_size + self.lambda_L2*self.W 
        loss= np.mean(L)  + 0.5*self.lambda_L2*L2

        return loss, dW
