import numpy as np 
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras import losses, activations, regularizers

from keras.optimizers import SGD

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})


def plot_ejercicio(history, label="", color='black'):
    acc = np.array(history.history['binary_accuracy'])
    epocas = np.arange(len(acc))
    plt.figure(1)
    plt.ylabel("Precisión [%]")
    plt.xlabel("Épocas")
    plt.plot(epocas, 100*acc, label=label, c=color, alpha=0.6)
    
    plt.legend(loc=0)
    
    plt.figure(2)
    plt.ylabel("Pérdida")
    plt.xlabel("Épocas")
    plt.plot(epocas, history.history['loss'], label=label, c=color, alpha=0.6)
    
    plt.legend(loc=0)
    

def ejer6_221(x_train, y_train, lr):
    model = models.Sequential()
    model.add(layers.Dense(2, 
                           input_dim=2, 
                           activation='tanh', 
                           use_bias=True, 
                           bias_initializer='random_uniform'))
    
    model.add(layers.Dense(1, 
                           activation='tanh', 
                           use_bias=True, 
                           bias_initializer='random_uniform'))

    sgd = optimizers.SGD(lr=lr)

    model.compile(loss='mean_squared_error', 
                  optimizer=sgd, 
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    
    history = model.fit(x_train, y_train, batch_size=1, epochs=250, verbose=0)
	
    plot_ejercicio(history, "Entrenamiento (221)", "red")
 


def ejer6_211(x_train, y_train, lr):
    inp1 = layers.Input(shape=(2,))
    
    x = layers.Dense(1, activation='tanh', 
                     use_bias=True)(inp1)
    x = layers.Concatenate()([inp1, x])
    
    output = layers.Dense(1, activation='tanh', 
                          use_bias=True)(x)
    
    model = models.Model(inputs=inp1, outputs=output) 
    #model.summary()
    sgd = optimizers.SGD(lr=lr)

    model.compile(loss='mean_squared_error', 
                  optimizer=sgd, 
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    
    history = model.fit(x_train, y_train, batch_size=1, epochs=250, verbose=0)
    
    plot_ejercicio(history, "Entrenamiento (211)", "blue")

def main():
    x_train = np.array([[0, 0],  [1, 0], [0, 1], [1, 1],])
    y_train = np.array([ [1],     [0],    [0],     [1]])

    lr =0.1
    ejer6_221(x_train, y_train, lr)
    ejer6_211(x_train, y_train, lr)
    
    plt.figure(1)
    plt.savefig("../docs/Figs/ejer2_6_acc.pdf")
    plt.figure(2)
    plt.savefig("../docs/Figs/ejer2_6_los.pdf")
    
    plt.show()

if __name__ == '__main__':
    main()
    