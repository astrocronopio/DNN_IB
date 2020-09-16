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

def run_cifar10():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train = x_train.astype(np.float)


    # transformo a la imagen a un vector unidimensional
    X = np.copy(x_train) 
    X = np.reshape(X, (X.shape[0], np.prod(x.shape[1:])))
    X/=255 #Porque las imagenes del cifar10 y mnist varian hasta 255

    X_test=np.copy(x_test) 
    X_test= np.reshape(X_test, (X_test.shape[0], np.prod(x_test.shape[1:])))
    X_test=X_test/255

    print('Dimensiones del set de entrenamiento ', x_train.shape)
    print(x_train.shape[0], 'ejemplos de entrenamiento')
    print(x_test.shape[0], 'ejemplos para probar')

    model_SVM = SVM(eta=0.0001, epochs = 200, batch_size=50, use_bias=True, lambda_L2=0.1)
    model_SVM.fit(X, y_train[:], X_test[:], y_test[:]) 
    #print("Con el test dado: ", model_SVM.error_pres," de ",len(y_test)," bien" )

    model_SMC = SMC(eta=0.0001, epochs = 200, batch_size=50, use_bias=True, lambda_L2=0.1)
    model_SMC.fit(X_train[:], y_train[:], X_test[:], y_test[:])   
    #print("Con el test dado: ", model_SMC.error_pres," de ",len(y_test)," bien" )
   
    return model_SMC, model_SVM


def run_mnist():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.astype(np.float)

    # transformo a la imagen a un vector unidimensional
    X = np.copy(x_train) 
    X = np.reshape(X, (X.shape[0], np.prod(x.shape[1:])))
    X/=255 #Porque las imagenes del cifar10 y mnist varian hasta 255

    X_test=np.copy(x_test) 
    X_test= np.reshape(X_test, (X_test.shape[0], np.prod(x_test.shape[1:])))
    X_test=X_test/255


    print('Dimensiones del set de entrenamiento ', x_train.shape)
    print(x_train.shape[0], 'ejemplos de entrenamiento')
    print(x_test.shape[0], 'ejemplos para probar')

    model_SVM = SVM(eta=0.0001, epochs = 200, batch_size=32, use_bias=True, lambda_L2=0.1)
    model_SVM.fit(X_train[:], y_train[:], X_test[:], y_test[:]) 
    #print("Con el test dado: ", model_SVM.error_pres," de ",len(y_test)," bien" )

    model_SMC = SMC(eta=0.0001, epochs = 200, batch_size=32, use_bias=True, lambda_L2=0.1)
    model_SMC.fit(X_train[:], y_train[:], X_test[:], y_test[:])   
    #print("Con el test dado: ", model_SMC.error_pres," de ",len(y_test)," bien" )

    return model_SMC, model_SVM

def ejer5():

    choose = int(input(u"Ingrese (1) para el MNIST y (2) para el CIFAR-10: "))

    if(choose==1): 
        title="MNIST"
        print("\n___Trabajando con ",title,"___")
        SMC_m, SVM_m= run_mnist()
        
    elif (choose==2): 
        title="CIFAR-10"
        print("\n___Trabajando con CIFAR-10.___\n(Es una RAM eater)")
        SMC_m, SVM_m= run_cifar10()        
    else: print("No entendí.")


    plt.figure(1)
    plt.ylabel('loss')
    plt.xlabel("Épocas")
    plt.yscale('log')
    plt.title(title)
    plt.plot(SVM_m.error_loss,color="blue", alpha=0.6, label="Support Vector Machine")
    plt.plot(SMC_m.error_loss,color="red", alpha=0.6, label="SoftMax Classifier")
    plt.legend(loc=0)
    plt.savefig("ejer_5_"+title+"_los.pdf")

    plt.figure(2)
    plt.xlabel("Épocas")
    plt.ylabel('accuracy')
    plt.title(title)
    #plt.axhline(100, color='black', alpha=0.5, ls='--')
    plt.plot(SVM_m.error_acc,color="blue",alpha=0.6, label="Support Vector Machine")
    plt.plot(SMC_m.error_acc,color="red", alpha=0.6, label="SoftMax Classifier")
    plt.plot(SVM_m.error_pres,color="blue", ls='--',  label="SVM - Test {:4.3}%".format(SVM_m.error_pres[-1]))
    plt.plot(SMC_m.error_pres,color="red", ls='--', label="SMC - Test: {:4.3}%".format(SMC_m.error_pres[-1]))
    plt.legend(loc=0)
    plt.savefig("ejer_5_"+title+"_acc.pdf")
    
    plt.show()

def main():
    ejer5()
    pass

if __name__ == '__main__':
    main()
    