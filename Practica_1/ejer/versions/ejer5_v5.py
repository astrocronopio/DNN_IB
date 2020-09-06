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

def run_vsm(x_train, y_train, x_test, y_test, eta):
    model_SVM = SVM(eta=eta, epochs = 1200, batch_size=50, use_bias=True, lambda_L2=0.05)
    model_SVM.fit(x_train, y_train, x_test, y_test) 
    print("Con el test dado: ", model_SVM.error_pres," de ",len(y_test)," bien" )
    return model_SVM.error_acc,model_SVM.error_loss,model_SVM.error_pres
    

def run_smc(x_train, y_train, x_test, y_test, eta):
    model_SMC = SMC(eta=eta, epochs = 1200, batch_size=50, use_bias=True, lambda_L2=0.05)
    model_SMC.fit(x_train, y_train, x_test, y_test)   
    print("Con el test dado: ", model_SMC.error_pres," de ",len(y_test)," bien" )
    return model_SMC.error_acc,model_SMC.error_loss, model_SMC.error_pres



def ejer_data(data,eta=0.05):
    (x_train, y_train), (x_test, y_test) = data.load_data()
    x_train = x_train.astype(np.float)
    print('Dimensiones del set de entrenamiento ', x_train.shape)
    print(x_train.shape[0], 'ejemplos de entrenamiento')
    print(x_test.shape[0], 'ejemplos para probar')
    
    acc_svm, loss_svm, pres_svm  = run_vsm(x_train, y_train, x_test, y_test,eta)
    acc_smc, loss_smc, pres_smc  = run_smc(x_train, y_train, x_test, y_test,eta)

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
    plt.ylabel('accuracy')
    plt.plot(100*np.ones_like(acc_svm), color='black', alpha=0.5, ls='--')
    plt.plot(acc_svm,color="blue",alpha=0.6, label="Support Vector Machine")
    plt.plot(acc_smc,color="red", alpha=0.6, label="SoftMax Classifier")
    plt.plot(pres_smc*np.ones_like(acc_svm),color="red", ls='--', label="SMC - Test: {:4.3}%".format(pres_smc))
    plt.plot(pres_svm*np.ones_like(acc_svm),color="blue", ls='--',  label="SVM - Test {:4.3}%".format(pres_svm))
    
    plt.legend(loc=0)
    plt.show()

def ejer5():
    ejer_data(datasets.mnist, eta=0.001)
    #ejer_data(datasets.mnist)

def main():
    ejer5()
    pass

if __name__ == '__main__':
    main()
    