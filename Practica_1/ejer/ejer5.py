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

def flattening(x_train,y_train):
    X = x_train #Me aseguro no modificar nunca los datos
    X = np.reshape(X, (X.shape[0], np.prod(X.shape[1:])))
    X=X/255
    Y = y_train.reshape(X.shape[0])
    X= np.hstack(( (np.ones((len(X),1) )), X)) 
    
    return X,Y

def run_fit(dataset):
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x_train = x_train.astype(np.float)

    # transformo a la imagen a un vector unidimensional
    X   ,Y  = flattening(x_train, y_train)
    X_t ,Y_t= flattening(x_test, y_test)

    print('Dimensiones del set de entrenamiento ', x_train.shape)
    print(x_train.shape[0], 'ejemplos de entrenamiento')
    print(x_test.shape[0], 'ejemplos para probar')

    print("\n====SoftMax====")
    model_SMC = SMC(eta=0.0002, epochs = 400, batch_size=50, lambda_L2=0.001)
    model_SMC.fit(X, Y, X_t, Y_t)   
   
    print("\n==Support Vector Machine==")
    model_SVM = SVM(eta=0.0002, epochs = 400, batch_size=50, lambda_L2=0.001)
    model_SVM.fit(X, Y, X_t, Y_t) 

    return model_SMC, model_SVM

def ejer5():

    choose = int(input(u"Ingrese (1) para el MNIST y (2) para el CIFAR-10: "))

    if(choose==1): 
        title="MNIST"
        print("\n___Trabajando con ",title,"___")
        SMC_m, SVM_m= run_fit(datasets.mnist)
        
    elif (choose==2): 
        title="CIFAR-10"
        print("\n___Trabajando con CIFAR-10.___\n(Es una RAM eater)")
        SMC_m, SVM_m= run_fit(datasets.cifar10)        
    else: print("No entendí.")

    outputfile = "ejer5_"+title+"_v2.dat"

    #np.save(outputfile, )
    
    np.savetxt(outputfile,   np.array([SVM_m.error_loss,
                             SMC_m.error_loss,
                             SVM_m.loss_test,
                             SMC_m.loss_test,
                             SVM_m.error_acc,
                             SMC_m.error_acc,
                             SVM_m.error_pres,
                             SMC_m.error_pres]).T)

    plt.figure(1)
    plt.ylabel('loss')
    plt.xlabel("Épocas")
    #plt.yscale('log')
    plt.title(title)
    plt.plot(SVM_m.error_loss,color="blue",   ls='--', alpha=0.6, label="SVM - Train")
    plt.plot(SMC_m.error_loss,color="red",   ls='--', alpha=0.6, label="SMC - Train")
    plt.plot(SVM_m.loss_test,color="blue",  label="SVM - Test")
    plt.plot(SMC_m.loss_test,color="red", label="SMC - Test")
    
    plt.legend(loc=0)
    plt.savefig("ejer_5_"+title+"_los.pdf")

    plt.figure(2)
    plt.xlabel("Épocas")
    plt.ylabel('accuracy [%]')
    
    plt.title(title)
    plt.plot(SVM_m.error_acc,   color="blue",   ls='--',   alpha=0.6, label="SVM - Train".format(SVM_m.error_acc[-1]))
    plt.plot(SMC_m.error_acc,   color="red",   ls='--',    alpha=0.6, label="SMC - Train".format(SMC_m.error_acc[-1]))
    plt.plot(SVM_m.error_pres,  color="blue", label="SVM - Test {:4.3}%".format(SVM_m.error_pres[-1]))
    plt.plot(SMC_m.error_pres,  color="red",   label="SMC - Test: {:4.3}%".format(SMC_m.error_pres[-1]))
    
    plt.legend(loc=0)
    plt.savefig("ejer_5_"+title+"_acc.pdf")
    
    plt.show()

def main():
    ejer5()
    pass

if __name__ == '__main__':
    main()