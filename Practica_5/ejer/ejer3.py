# Proponga un ejemplo para aplicar los conceptos de transferencia de aprendizaje
# y discuta si los resultados obtenidos eran los esperados. ¿Cuándo se puede 
# esperar que este tipo de técnicas funciones bien, y cuándo no?

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import MobileNet, mobilenet
from tensorflow.keras import losses, optimizers, metrics

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from imageio import  imread
from os import listdir

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from random import shuffle

import tensorflow.keras as K
import tensorflow as tf

def plot_ejercicio(history):   
     
    acc_train = 100*np.array(history.history['acc'])
    acc_test  = 100*np.array(history.history['val_acc'])

    loss = np.array(history.history['loss'])
    val_loss  = np.array(history.history['val_loss'])  
    
    return acc_train, acc_test, loss, val_loss

def preprocessing(X, Y):
    X = mobilenet.preprocess_input(X)
    Y = to_categorical(Y, 10)
    return X, Y


def ejer3_fine_tuning(n_epochs):
    input_image = Input(shape=(32, 32, 3))
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, y_train = preprocessing(x_train, y_train)
    x_test, y_test = preprocessing(x_test, y_test)
    
    model_keras = MobileNet( include_top=False,
                            weights="imagenet",
                            input_tensor=input_image)
    model_keras.trainable = False

    model = Sequential()
    model.add(model_keras)
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss=losses.CategoricalCrossentropy(), 
                  optimizer=optimizers.Adam(learning_rate=0.0001), 
                  metrics=[metrics.CategoricalAccuracy('acc')]) 
    
    model.summary()
    
    history = model.fit(
                        x_train, y_train,
                        validation_data=(x_test, y_test),
                        batch_size=50,
                        epochs=20,
                        verbose=1
                        )
    
    acc, val_acc, loss, val_loss =  plot_ejercicio(history)  
    
    ACC = np.array([])
    VAL_ACC =  np.array([])
    LOSS = np.array([])
    VAL_LOSS =  np.array([])
    
    
    ACC = np.append(ACC, acc)
    VAL_ACC = np.append(VAL_ACC, val_acc)
    LOSS = np.append(LOSS, loss)
    VAL_LOSS = np.append(VAL_LOSS, val_loss)
    
    
    model_keras.trainable = True
    
    model.compile(loss=losses.CategoricalCrossentropy(), 
                  optimizer=optimizers.Adam(learning_rate=0.00001), 
                  metrics=[metrics.CategoricalAccuracy('acc')]) 
    
    model.summary()
    
    history = model.fit(
                        x_train, y_train,
                        validation_data=(x_test, y_test),
                        batch_size=50,
                        epochs=n_epochs,
                        verbose=1
                        )
    
    
    
    acc, val_acc, loss, val_loss =  plot_ejercicio(history)  
    
    ACC = np.append(ACC, acc)
    VAL_ACC = np.append(VAL_ACC, val_acc)
    LOSS = np.append(LOSS, loss)
    VAL_LOSS = np.append(VAL_LOSS, val_loss)
    
    np.savetxt("ejer3{}epochs{}_mobilenet.txt".format("_fine_tuning_", n_epochs),
                np.array([ ACC, VAL_ACC, LOSS, VAL_LOSS ]).T)
    
    
def ejer3_none(n_epochs):
    input_image = Input(shape=(32, 32, 3))
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, y_train = preprocessing(x_train, y_train)
    x_test, y_test = preprocessing(x_test, y_test)
    
    model_keras = MobileNet( include_top=False,
                            weights=None,
                            input_tensor=input_image)
    #model_keras.trainable = False

    model = Sequential()
    model.add(model_keras)
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss=losses.CategoricalCrossentropy(), 
                  optimizer=optimizers.Adam(learning_rate=0.00001), 
                  metrics=[metrics.CategoricalAccuracy('acc')]) 
    
    model.summary()
    
    history = model.fit(
                        x_train, y_train,
                        validation_data=(x_test, y_test),
                        batch_size=50,
                        epochs=n_epochs,
                        verbose=1
                        )
    
    acc, val_acc, loss, val_loss =  plot_ejercicio(history)    
    
    np.savetxt("ejer3{}epochs{}_mobilenet.txt".format("_none_", n_epochs),
                np.array([ acc, val_acc, loss, val_loss ]).T)
    
    
def ejer3_imagenet(n_epochs):
    input_image = Input(shape=(32, 32, 3))
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, y_train = preprocessing(x_train, y_train)
    x_test, y_test = preprocessing(x_test, y_test)
    
    model_keras = MobileNet( include_top=False,
                            weights="imagenet",
                            input_tensor=input_image)
    #model_keras.trainable = False

    model = Sequential()
    model.add(model_keras)
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss=losses.CategoricalCrossentropy(), 
                  optimizer=optimizers.Adam(learning_rate=0.00001), 
                  metrics=[metrics.CategoricalAccuracy('acc')]) 
    
    model.summary()
    
    history = model.fit(
                        x_train, y_train,
                        validation_data=(x_test, y_test),
                        batch_size=50,
                        epochs=n_epochs,
                        verbose=1
                        )
    
    acc, val_acc, loss, val_loss =  plot_ejercicio(history)    
    
    np.savetxt("ejer3{}epochs{}_mobilenet.txt".format("_imagenet_", n_epochs),
                np.array([ acc, val_acc, loss, val_loss ]).T)
        
    

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})
cmap = plt.get_cmap('viridis_r',9)
    
def plot_ejer(n_epochs):
    acc_1, val_acc_1, loss_1, val_loss_1 = np.loadtxt("./ejer3_none_epochs100_mobilenet.txt", unpack=True)
    acc_2, val_acc_2, loss_2, val_loss_2 = np.loadtxt("ejer3_imagenet_epochs100_mobilenet.txt", unpack=True)
    acc_3, val_acc_3, loss_3, val_loss_3 = np.loadtxt("./ejer3_fine_tuning_epochs80_mobilenet.txt", unpack=True)
    
    
    plt.figure(1)
    plt.xlabel("Épocas")
    plt.xlim(right=60)
    plt.ylabel("Precisión [%]")
    plt.plot(acc_1  , label="E",        c=cmap(5) , alpha=0.9, ls='--')
    plt.plot(val_acc_1, label="V",        c=cmap(4), alpha=0.9)
    
    plt.plot(acc_2  , label="E-Imag.",   c=cmap(3), alpha=0.9, ls='--')
    plt.plot(val_acc_2, label="V-Imag.",   c=cmap(2) , alpha=0.9)

    plt.plot(acc_3  , label="E-FT", c=cmap(1), alpha=0.9, ls='--')
    plt.plot(val_acc_3, label="V-FT", c=cmap(0) , alpha=0.9)    
    
    plt.legend(loc=0, ncol=3)
    plt.savefig("../docs/Figs/ejer3_acc.pdf")
    
    plt.figure(2)
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida Normalizada")
    plt.xlim(right=60)
    
    plt.plot(loss_1/np.max(loss_1)         , label="E", c=cmap(5), alpha=0.9, ls='--')
    plt.plot(val_loss_1/np.max(val_loss_1) , label="V", c=cmap(4), alpha=0.9)
    
    plt.plot(loss_2 / np.max(loss_2)        , label="E-Imag.", c=cmap(3), alpha=0.9, ls='--')
    plt.plot(val_loss_2 / np.max(val_loss_2), label="V-Imag.", c=cmap(2), alpha=0.9)

    plt.plot(loss_3 / np.max(loss_3)        , label="E-FT", c=cmap(1), alpha=0.9, ls='--')
    plt.plot(val_loss_3 / np.max(val_loss_3), label="V-FT", c=cmap(0), alpha=0.9)
    
    plt.legend(loc=0, ncol=3)
    plt.savefig("../docs/Figs/ejer3_loss.pdf")

    plt.show()
 
if __name__ == "__main__":
    #ejer3_fine_tuning(80)
    #ejer3_imagenet(100)
    #ejer3_none(100)
    
    plot_ejer(2)