#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np 
import tensorflow.keras.datasets  as datasets
from classifier import LinearClassifier, SVM, SMC

import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})

def ejer5():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.astype(np.float)
    print('Dimensiones del set de entrenamiento ', x_train.shape)
    print(x_train.shape[0], 'ejemplos de entrenamiento')
    print(x_test.shape[0], 'ejemplos para probar')

    model_SVM = SVM(eta=0.01, epochs = 1000, batch_size=100, use_bias=True)
    model_SVM.fit(x_train, y_train) 

    #model_SMC = SMC(eta=0.01, epochs = 1, batch_size=2, use_bias=True)
    #model_SMC.fit(x_train, y_train)   

   

    plt.figure(1)
    plt.ylabel('loss')
    plt.xlabel("Épocas")
    plt.yscale('log')
    plt.plot(model_SVM.error_loss)
    
    plt.figure(2)
    plt.xlabel("Épocas")
    plt.ylabel('acc')
    plt.plot(100*np.ones_like(model_SVM.error_acc))
    plt.plot(model_SVM.error_acc)
    plt.show()
 
    

""" 
    model_SMC = SMC(eta=0.1, epochs = 100)
    error_loss_2, error_acc_2= model_SMC.fit(x_train[:5], y_train[:5])  

    plt.figure(1)
    plt.title('loss')
    plt.plot(error_loss_1)
    plt.plot(error_loss_2)

    plt.figure(2)
    plt.title('acc')
    plt.plot(error_acc_1)
    plt.plot(error_acc_2)

    plt.show() """
    
def main():
    ejer5()
    pass

if __name__ == '__main__':
    main()
    