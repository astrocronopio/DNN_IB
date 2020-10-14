import pandas 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras import  models, layers, optimizers
from tensorflow.keras import  losses, activations, regularizers

import os
from sklearn.model_selection import KFold, StratifiedKFold

import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})


def data_loading():
    df = pandas.read_csv("~/Desktop/Datos/pima-indians-diabetes.csv", header=None)
    df = df.to_numpy()
    x, y = df[:,0:8:1], df[:,8]
    return x,y

def preprocessing(x_train, x_test):
    media = x_train.mean(axis=0)
    sigma = x_train.std(axis=0)

    x_train -= media
    x_train /= sigma  

    x_test  -= media
    x_test  /= sigma
    
    return x_train, x_test

def model_definition(input_shape):
    model = models.Sequential()

    model.add(layers.Dense(
                    units=10, 
                    input_shape=(input_shape,), 
                    use_bias=True,
                    activation=activations.relu,
                    activity_regularizer=regularizers.L2(0.000)))

    # model.add(layers.Dense(
    #                 units=3,
    #                 activity_regularizer=regularizers.L2(0.000),
    #                 activation=activations.relu,
    #                 use_bias=True))

    model.add(layers.Dense(
                    units=1,
                    activity_regularizer=regularizers.L2(0.000),
                    activation=activations.relu,
                    use_bias=True))

    model.compile(  
                    optimizer=optimizers.SGD(0.0005),
                    loss=losses.MSE, 
                    metrics=['acc'])
    
    return model
    
def ejer6():
    n_epochs=500
    n_splits=5
    kf = KFold(n_splits = 5)

    val_acc_kfold = []#np.zeros(size=(n_epochs, n_splits))
    val_loss_kfold = []#np.zeros(size=(n_epochs, n_splits))

    x,y = data_loading()


    for train_index, test_index in kf.split(x):
        x_train, x_test  = x[train_index], x[test_index]
        y_train, y_test  = y[train_index], y[test_index]
        
        x_train, x_test = preprocessing(x_train, x_test)

        model = model_definition(x_train.shape[1])
        
        history = model.fit(
                        x_train, y_train, 
                        epochs=n_epochs, 
                        batch_size=5,
                        validation_data=(x_test,y_test))
        
        val_acc_kfold.append(history.history['val_acc'])
        val_loss_kfold.append(history.history['val_loss'])
        
        tf.keras.backend.clear_session()  #Resetea todo  
        
    val_acc_kfold = np.array(val_acc_kfold)
    val_loss_kfold = np.array(val_loss_kfold)
        
    plt.figure(1)
    plt.ylabel("Precisión [%]")
    plt.plot(np.mean(val_acc_kfold, axis=0), label="Validación", c='blue', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer6_acc.pdf")

    plt.figure(2)
    plt.ylabel("Pérdida")
    plt.plot(np.mean(val_loss_kfold, axis=0), label="Validación", c='blue', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer6_loss.pdf")

    plt.show()
    
if __name__ == '__main__':
    ejer6()
    