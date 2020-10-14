"""
Versión 1: Con k-fold, todavía no coverge plot

"""


import pandas 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras import  models, layers, optimizers, losses, activations, regularizers

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

def get_model_name(k):
    return 'model_'+str(k)+'.h5'


kf = KFold(n_splits = 5)

VALIDATION_ACCURACY = []
VALIDAITON_LOSS = []
                         

save_dir = '/saved_models/'
fold_var = 1

x,y = data_loading()

for train_index, test_index in kf.split(x):
    x_train, x_test  = x[train_index], x[test_index]
    y_train, y_test  = y[train_index], y[test_index]
    

    model = models.Sequential()

    model.add(layers.Dense(
                    units=8, 
                    input_shape=(x_train.shape[1],), 
                    use_bias=True,
                    activation=activations.relu,
                    activity_regularizer=regularizers.L2(0.000)))

    model.add(layers.Dense(
                    units=5,
                    activity_regularizer=regularizers.L2(0.000),
                    activation=activations.relu,
                    use_bias=True))

    model.add(layers.Dense(
                    units=1,
                    activity_regularizer=regularizers.L2(0.000),
                    activation=activations.linear,
                    use_bias=True))

    model.compile(  
                    optimizer=optimizers.SGD(0.001),
                    loss=losses.MSE, 
                    metrics=['acc'])

    history = model.fit(
                    x_train, y_train, 
                    epochs=100, 
                    batch_size=5,
                    validation_data=(x_test,y_test))
    
    VALIDATION_ACCURACY.append(history.history['val_acc'])
    VALIDAITON_LOSS.append(history.history['val_loss'])

    # plt.figure(1)
    # plt.ylabel("Precisión [%]")
    # plt.plot(100*history.history['acc']    , label="Entrenamiento", c='red', alpha=0.6, ls='--')
    # plt.plot(100*history.history['val_acc'], label="Validación", c='blue', alpha=0.6)
    # plt.legend(loc=0)
    # #plt.savefig("../docs/Figs/ejer3_acc.pdf")

    # plt.figure(2)
    # plt.ylabel("Pérdida")
    # plt.plot(history.history['loss']    , label="Entrenamiento", c='red', alpha=0.6, ls='--')
    # plt.plot(history.history['val_loss'], label="Validación", c='blue', alpha=0.6)
    # plt.legend(loc=0)
    # #plt.savefig("../docs/Figs/ejer3_loss.pdf")
    
    tf.keras.backend.clear_session()
    
plt.figure(1)
plt.ylabel("Precisión [%]")
#plt.plot(VALIDATION_ACCURACY.mean(axis=1), label="Entrenamiento", c='red', alpha=0.6, ls='--')
plt.plot(VALIDATION_ACCURACY.mean(axis=1), label="Validación", c='blue', alpha=0.6)
plt.legend(loc=0)
#plt.savefig("../docs/Figs/ejer3_acc.pdf")

plt.figure(2)
plt.ylabel("Pérdida")
#plt.plot(history.history['loss']    , label="Entrenamiento", c='red', alpha=0.6, ls='--')
plt.plot(VALIDAITON_LOSS.mean(axis=1), label="Validación", c='blue', alpha=0.6)
plt.legend(loc=0)
#plt.savefig("../docs/Figs/ejer3_loss.pdf")

plt.show()