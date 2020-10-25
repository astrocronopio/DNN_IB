from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import losses, activations, regularizers, optimizers

import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})

def permute_mnist(x_train, x_test):
    permutation = np.random.permutation(28*28)
    
    x_train_perm = x_train.reshape(x_train.shape[0], -1)
    x_train_perm = x_train_perm[:,permutation]
    x_train_perm = x_train_perm.reshape(x_train.shape)
    
    x_test_perm = x_test.reshape(x_test.shape[0], -1)
    x_test_perm = x_test_perm[:,permutation]
    x_test_perm = x_test_perm.reshape(x_test.shape)   
     
    return x_train_perm, x_test_perm
    


def preprocesing():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Reshaping las imagenes para que tengan un "color", ademas para hacerlos float
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32')
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32')
    
    x_train = x_train / 255.
    x_test = x_test / 255.
    
    y_train = to_categorical(y_train) #Como la guia anterior
    y_test = to_categorical(y_test)   #Para usar el categorical sin asco
    
    return (x_train, y_train), (x_test, y_test)

    
def model_definition_conv(num_classes):
    
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(10, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    #Le mande FC ya fue
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizers.Adam(learning_rate=0.0001), 
                  metrics=['acc'])
    model.summary()
    return model

def ejer9():
    (x_train, y_train), (x_test, y_test) = preprocesing()
    

    
    num_classes = y_test.shape[1]

    model_conv1= model_definition_conv(num_classes)
    model_conv2 = model_definition_conv(num_classes)
    #exit()
    
    
    history_conv1= model_conv1.fit(x_train, y_train, 
                        epochs=50, 
                        batch_size=100,
                        validation_data=(x_test, y_test))

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
    x_train, x_test = permute_mnist(x_train, x_test)
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
    
    history_conv2 = model_conv2.fit(x_train, y_train, 
                        epochs=50, 
                        batch_size=100,
                        validation_data=(x_test, y_test))

    acc_conv1, val_acc_conv1, loss_conv1, val_loss_conv1= plot_ejercicio(history_conv1)
    acc_conv2, val_acc_conv2, loss_conv2, val_loss_conv2 = plot_ejercicio(history_conv2)
  
    
    plt.figure(1)
    plt.xlabel("Épocas")
    plt.ylabel("Precisión [%]")
    plt.plot(acc_conv1  , label="Train Conv. ", c='orange', alpha=0.6, ls='--')
    plt.plot(acc_conv2 , label="Train Conv. Perm.", c='green', alpha=0.6, ls='--')
    
    plt.plot(val_acc_conv1, label="Test Conv. ", c='blue', alpha=0.6)    
    plt.plot(val_acc_conv2, label="Test Conv. Perm.", c='red', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer9_acc.pdf")
    
    plt.figure(2)
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida Normalizada")
    plt.plot(loss_conv1    /np.max(loss_conv1    ), label="Train Conv. ", c='orange', alpha=0.6, ls='--')
    plt.plot(loss_conv2    /np.max(loss_conv2    ), label="Train Conv. Perm.", c='green', alpha=0.6, ls='--')
    
    plt.plot(val_loss_conv1/np.max(val_loss_conv1), label="Test Conv. ", c='blue', alpha=0.6)
    plt.plot(val_loss_conv2/np.max(val_loss_conv2), label="Test Conv. Perm.", c='red', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer9_loss.pdf")

    plt.show()

def plot_ejercicio(history):   
     
    acc_train = 100*np.array(history.history['acc'])
    acc_test  = 100*np.array(history.history['val_acc'])

    loss = np.array(history.history['loss'])
    val_loss  = np.array(history.history['val_loss'])  
    
    return acc_train, acc_test, loss, val_loss


if __name__ == '__main__':
    ejer9()
    