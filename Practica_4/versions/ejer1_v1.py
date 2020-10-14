"""
Versión 1: Con k-fold, todavía no coverge plot
Versión 2: This
"""

import tensorflow as tf 

from tensorflow.keras import models, layers, optimizers, losses
import numpy as np 

from sklearn.datasets import load_boston  


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})


def boston_data():        
    X, Y = load_boston(return_X_y=True)
    
    idx = np.arange(X.shape[0])
    idx25 = int(X.shape[0]/4)
    np.random.shuffle(idx)

    x_train, y_train = X[idx[idx25:]], Y[idx[idx25:]]
    x_test , y_test  = X[idx[:idx25]], Y[idx[:idx25]]
    
    return x_train, y_train, x_test, y_test


def ejer1():
    x_train, y_train, x_test, y_test = boston_data()
    
    media = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)

    x_train -= media
    x_train /= sigma  
    
    x_test  -= media
    x_test  /= sigma

    model = models.Sequential()

    model.add(layers.Dense(units=1, 
                        input_shape=(13,), 
                        use_bias=True))

    model.compile(optimizer=optimizers.SGD(0.001),
                loss=losses.MSE, 
                metrics=['mse'])

    history = model.fit(x_train, y_train, 
                        epochs=50, 
                        batch_size=2,
                        validation_data=(x_test,y_test))


    loss_train = history.history['mse']
    loss_test  = history.history['val_mse']

    plt.figure(2)
    plt.ylabel("Precios [1000 USD]")
    plt.xlabel("Cantidad de personas en la clase baja [%]")
    X_TEST = (x_test)*sigma + media
    
    plt.scatter(X_TEST[:,12], y_test, c='black', alpha=0.5, label="Datos" , marker='*', s=100)
    plt.scatter(X_TEST[:,12], model.predict(x_test), c='red', alpha=0.7, label="Predicción")
    plt.legend(loc=0)
    plt.savefig("ejer1_low_income.pdf")
    

    plt.figure(4)
    plt.ylabel("Datos de precios  [1000 USD]")
    plt.xlabel("Predición de precios  [1000 USD]")
    plt.scatter(y_test, model.predict(x_test), c='green', alpha=0.6)
    plt.plot(y_test, y_test, c='black', alpha=0.8, label="Referencia")#plt.plot(y_test)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer1_versus.pdf")
    
    
    plt.figure(1)
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.plot(loss_train,label="Entrenamiento", c='red', alpha=0.6)
    plt.plot(loss_test, label="Validación", c='blue', alpha=0.6)
    plt.savefig("../docs/Figs/ejer1_loss.pdf")
    plt.show()
    # x_max = X.max()
    # y_max = Y.max()

    # print(X[:,1].shape, Y.shape)

    # x_train, y_train = X/x_max, Y/y_max


    # plt.figure(3)
    # plt.scatter(X[:,12], Y)

    # plt.show()

    # exit()
    
if __name__ == '__main__':
    ejer1()