import pandas 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras import  models, layers, optimizers
from tensorflow.keras import  losses, activations, regularizers

"""
Versión 1: Con k-fold, todavía no coverge plot
Versión 2: This
"""

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
    
    media_paper = np.array([    
      3.8 , #Pregnant
    120.9 , #Glucose Complete
     69.1 , #Blood Complete
     20.5 , #Triceps
     79.8 , #Insulin
     32.0 , #BMI
      0.5 , #Diabetes
     33.2 ]) #Age
    
    # for paciente in x:
    #     if paciente[1]==0.0: #Glucose Complete
    #         paciente[1]= media_paper[1]

    #     if paciente[2]==0.0: #Blood Complete
    #         paciente[2]= media_paper[2]

    #     if paciente[4]==0.0:#Insulin
    #         paciente[4]= media_paper[4]

    #     if paciente[5]==0.0: #BMI
    #         paciente[5]= media_paper[5]

    return x,y, 

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
                    optimizer=optimizers.SGD(0.001),
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
    
    min_acc = np.min(val_acc_kfold, axis=0)
    max_acc = np.max(val_acc_kfold, axis=0)
    
    min_loss = np.min(val_loss_kfold, axis=0)
    max_loss = np.max(val_loss_kfold, axis=0)
    
    epocas = np.arange(len(min_acc))    
    plt.figure(1)
    plt.ylabel("Precisión [%]")
    plt.xlabel("Épocas")
    plt.plot(epocas, np.mean(100*val_acc_kfold, axis=0), label="Validación", c='blue', alpha=0.6)
    plt.fill_between(epocas, 100*max_acc, 100*min_acc, alpha=0.1)
    
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer6_acc.pdf")

    plt.figure(2)
    plt.ylabel("Pérdida")
    plt.xlabel("Épocas")
    plt.plot(epocas, np.mean(val_loss_kfold, axis=0), label="Validación", c='blue', alpha=0.6)
    plt.fill_between(epocas, max_loss, min_loss, alpha=0.1)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer6_loss.pdf")

    plt.show()
    
if __name__ == '__main__':
    ejer6()
    