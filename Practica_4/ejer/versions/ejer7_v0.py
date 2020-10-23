"""
Finalmente, el famoso autoencoder
"""
  
from tensorflow.keras.datasets import mnist
from tensorflow.keras import  models, layers, optimizers
from tensorflow.keras import  losses, activations, regularizers

import numpy as np
import tensorflow as tf


import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [10, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})

def preprocesing():
    (x_train, _), (x_test, y_name) = mnist.load_data()
    
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    
    y_train= np.copy(x_train)
    y_test= np.copy(x_test)
        
    x_train += np.random.normal(0,0.5, size=x_train.shape)#[np.newaxis,:]
    x_test += np.random.normal(0,0.5, size=x_test.shape)#[np.newaxis,:]

    x_train = np.clip(x_train, a_min=0, a_max=1)
    x_test = np.clip(x_test, a_min=0, a_max=1)
    
    return x_train, x_test, y_train , y_test  ,y_name

def ejer7():
    x_train, x_test, y_train , y_test  ,y_name = preprocesing()

    encoding_dim = 32  # Encoded image 32x32
    raw_dim=784

    input_img =layers.Input(shape=(raw_dim,)) # Raw input

    encoded = layers.Dense(encoding_dim, 
                        activation='relu')(input_img) # Encoded output

    decoded = layers.Dense(raw_dim, 
                        activation='sigmoid')(encoded) # Decoded output

    # Raw --> encoded --> decoded
    autoencoder =models.Model(input_img, decoded)

    # Raw --> Encoded
    encoder =models.Model(input_img, encoded)

    ########################################################################
    encoded_input =layers.Input(shape=(encoding_dim,)) # Encoded input
    decoder_layer = autoencoder.layers[-1](encoded_input) # Decoded output

    # Encoded --> Decoded
    decoder =models.Model(encoded_input, decoder_layer)

    autoencoder.compile(optimizer=optimizers.SGD(0.1), loss=losses.BinaryCrossentropy())

    autoencoder.fit(x_train, x_train,
                    epochs=100,
                    batch_size=100,
                    shuffle=True,
                    validation_data=(x_test, x_test))


    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    plot_ejer7(x_test,y_name,decoded_imgs)


def plot_ejer7(x_test, y_test, decoded_imgs):
    """
    Así lo plotean en el ejemplos de keras.io
    """

    n = 3 
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        ax.set_title("Digito: {}".format(y_test[i]))
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig("../docs/Figs/ejer7.pdf")
    plt.show()



if __name__ == '__main__':
    ejer7()
    