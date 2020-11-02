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

def ejer3(n_epochs):
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
    #model.add(Dropout(0.4))
    #model.add(BatchNormalization())
    model.add(Dense(10, activation='softmax'))
    
    model.compile(loss=losses.CategoricalCrossentropy(), 
                  optimizer=optimizers.Adam(learning_rate=0.0001), 
                  metrics=[metrics.CategoricalAccuracy('acc')]) 
    
    model.summary()
    
    history = model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              batch_size=50,
              epochs=n_epochs,
              verbose=1)
    
    acc, val_acc, loss, val_loss =  plot_ejercicio(history)    
    
    np.savetxt("ejer3{}epochs{}_mobilenet.txt".format("_imagenet_fixed_", n_epochs),
                np.array([ acc, val_acc, loss, val_loss ]).T)
    
    model.save('cifar10_ejer3_imagenet_mobile_net.h5')


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})
    
def plot_ejer(n_epochs):
    acc, val_acc, loss, val_loss = np.loadtxt("ejer3{}epochs{}_mobilenet.txt".format("_imagenet_fixed_", n_epochs), unpack=True)

    
    plt.figure(1)
    plt.xlabel("Épocas")
    epocas = np.arange(len(acc))
    
    plt.ylabel("Precisión [%]")
    plt.plot(epocas, acc  , label="Train - Keras ", c='red', alpha=0.6, ls='--')
    plt.plot(epocas, val_acc, label="Test - Keras", c='blue', alpha=0.6)
    
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer3_acc.pdf")
    
    plt.figure(2)
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.plot(epocas, loss   , label="Train - Keras", c='red', alpha=0.6, ls='--')
    plt.plot(epocas, val_loss, label="Test - Keras", c='blue', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer3_loss.pdf")

    plt.show()
 
if __name__ == "__main__":
    ejer3(60)
    plot_ejer(60)