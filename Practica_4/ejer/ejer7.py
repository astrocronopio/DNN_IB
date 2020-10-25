"""
Finalmente, el famoso autoencoder
"""
  
from tensorflow.keras.datasets import mnist
from tensorflow.keras import  models, layers, optimizers
from tensorflow.keras import  losses, activations, metrics

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
    
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

    y_train= np.copy(x_train)
    y_test= np.copy(x_test)
        
    x_train += np.random.normal(0,0.5, size=x_train.shape)#[np.newaxis,:]
    x_test += np.random.normal(0,0.5, size=x_test.shape)#[np.newaxis,:]

    x_train = np.clip(x_train, a_min=0, a_max=1)
    x_test = np.clip(x_test, a_min=0, a_max=1)
    
    return x_train, x_test, y_train , y_test  ,y_name


def autoencoder_model():
    
    input_img = layers.Input(shape=(28, 28, 1))

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    #print(x.shape)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    #print(x.shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    #print(x.shape)
    x = layers.MaxPooling2D((2, 2), padding='same')(x) #Encoded
    #print(x.shape)
    
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    #print(x.shape)
    x = layers.UpSampling2D((2, 2))(x)
    #print(x.shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    #print(x.shape)
    x = layers.UpSampling2D((2, 2))(x)
    #print(x.shape)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    #print(decoded.shape)
    autoencoder = models.Model(input_img, decoded)

    ########################################################################

    autoencoder.compile(optimizer=optimizers.Adam(learning_rate=0.002),#optimizers.SGD(0.01), 
                        metrics=metrics.BinaryAccuracy(),
                        loss=losses.BinaryCrossentropy())

    return autoencoder


def ejer7():
    n_epochs=100
    x_train, x_test, y_train , y_test  ,y_name = preprocesing()

    autoencoder = autoencoder_model()
    
    autoencoder.summary()
    
    history = autoencoder.fit(x_train, y_train,
                    epochs=n_epochs,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, y_test))

    decoded_imgs = autoencoder.predict(x_test)

    plot_ejer7(x_test,y_name,decoded_imgs, history)


def plot_ejer7(x_test, y_test, decoded_imgs, history):
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

    plt.figure(5)
    plt.ylabel("Precisión [%]")
    
    acc= np.array( history.history['binary_accuracy'])
    val_acc =  np.array( history.history['val_binary_accuracy'])
    epocas = np.arange(len(acc))
    plt.plot(epocas, 100*acc    , label="Entrenamiento", c='red', alpha=0.6, ls='--')
    plt.plot(epocas, 100*val_acc, label="Validación", c='blue', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer7_acc.pdf")
    
    plt.figure(7)
    plt.ylabel("Pérdida")
    plt.plot(history.history['loss']    , label="Entrenamiento", c='red', alpha=0.6, ls='--')
    plt.plot(history.history['val_loss'], label="Validación", c='blue', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer7_loss.pdf")    
    
    
    plt.show()


if __name__ == '__main__':
    ejer7()
    