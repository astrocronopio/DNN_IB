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

''' def ejer5():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.astype(np.float)
    print('Dimensiones del set de entrenamiento ', x_train.shape)
    print(x_train.shape[0], 'ejemplos de entrenamiento')
    print(x_test.shape[0], 'ejemplos para probar')

    #np.random.seed(50)
    model_SMC = SMC(eta=0.1, epochs = 2000, batch_size=60, use_bias=True, lambda_L2=0.5)
    model_SMC.fit(x_train, y_train)   
    acc_smc, loss_smc=model_SMC.error_acc,model_SMC.error_loss
    
    del model_SMC
    #np.random.seed(40)

    model_SVM = SVM(eta=0.1, epochs = 2000, batch_size=60, use_bias=True, lambda_L2=0.5)
    model_SVM.fit(x_train, y_train) 
    acc_svm, loss_svm=model_SVM.error_acc,model_SVM.error_loss

    del model_SVM
     '''


def run_vsm():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.astype(np.float)
    print('Dimensiones del set de entrenamiento ', x_train.shape)
    print(x_train.shape[0], 'ejemplos de entrenamiento')
    print(x_test.shape[0], 'ejemplos para probar')
    
    model_SVM = SVM(eta=0.01, epochs = 2000, batch_size=60, use_bias=True, lambda_L2=0.5)
    model_SVM.fit(x_train, y_train) 
    return model_SVM.error_acc,model_SVM.error_loss
    

def run_smc():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.astype(np.float)
    print('Dimensiones del set de entrenamiento ', x_train.shape)
    print(x_train.shape[0], 'ejemplos de entrenamiento')
    print(x_test.shape[0], 'ejemplos para probar')
    
    model_SMC = SMC(eta=0.01, epochs = 5000, batch_size=60, use_bias=True, lambda_L2=0.5)
    model_SMC.fit(x_train, y_train)   

    return model_SMC.error_acc,model_SMC.error_loss


def ejer5():
    acc_svm, loss_svm = run_vsm()
    acc_smc, loss_smc = run_smc()

    plt.figure(1)
    plt.ylabel('loss')
    plt.xlabel("Épocas")
    plt.yscale('log')
    plt.plot(loss_svm,color="blue", alpha=0.6, label="Support Vector Machine")
    plt.plot(loss_smc,color="red", alpha=0.6, label="SoftMax Classifier")
    #plt.plot(np.zeros_like(model_SMC.error_acc), color='black', alpha=0.5, ls='--')
    plt.legend(loc=0)

    plt.figure(2)
    plt.xlabel("Épocas")
    plt.ylabel('acc')
    plt.plot(100*np.ones_like(acc_svm), color='black', alpha=0.5, ls='--')
    plt.plot(acc_svm,color="blue",alpha=0.6, label="Support Vector Machine")
    plt.plot(acc_smc,color="red", alpha=0.6, label="SoftMax Classifier")

    plt.legend(loc=0)
    plt.show()


def main():
    ejer5()
    pass

if __name__ == '__main__':
    main()
    