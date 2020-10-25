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

def model_definition_dense(num_classes, input_shape=784):
    model = Sequential()

    model.add(Dense(100, activation='relu', 
                    input_shape=(input_shape,),
                    activity_regularizer=regularizers.L2(0.1)))
    
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizers.SGD(lr=0.001), 
                  metrics=['acc'])    
    model.summary()
    return model
    
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
                  optimizer=optimizers.SGD(lr=0.001), 
                  metrics=['acc'])
    model.summary()
    return model

def ejer8():
    (x_train, y_train), (x_test, y_test) = preprocesing()
    

    num_classes = y_test.shape[1]

    model_conv = model_definition_conv(num_classes)
    model_dense = model_definition_dense(num_classes)
    #exit()
    
    
    history_conv = model_conv.fit(x_train, y_train, 
                        epochs=200, 
                        batch_size=100,
                        validation_data=(x_test, y_test))
    
    #@@@@@@@@@@@@@@@@@@@@@@@@- FLATTENING - @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
    x_train =  np.reshape(x_train, (x_train.shape[0], np.prod(x_train.shape[1:])))
    x_test =np.reshape(x_test, (x_test.shape[0], np.prod(x_test.shape[1:])))
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
    
    history_dense = model_dense.fit(x_train, y_train, 
                        epochs=200, 
                        batch_size=100,
                        validation_data=(x_test, y_test))

    acc_conv, val_acc_conv, loss_conv, val_loss_conv = plot_ejercicio(history_conv)
    acc_dense, val_acc_dense, loss_dense, val_loss_dense = plot_ejercicio(history_dense)
  
    
    plt.figure(1)
    plt.xlabel("Épocas")
    plt.ylabel("Precisión [%]")
    plt.plot(acc_dense  , label="Train Dense", c='orange', alpha=0.6, ls='--')
    plt.plot(acc_conv  , label="Train Conv.", c='green', alpha=0.6, ls='--')
    
    plt.plot(val_acc_dense, label="Test Dense", c='blue', alpha=0.6)    
    plt.plot(val_acc_conv, label="Test Conv.", c='red', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer8_acc.pdf")
    
    plt.figure(2)
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida Normalizada")
    plt.plot(loss_dense    /np.max(loss_dense    ), label="Train Dense", c='orange', alpha=0.6, ls='--')
    plt.plot(loss_conv     /np.max(loss_conv     ), label="Train Conv.", c='green', alpha=0.6, ls='--')
    
    plt.plot(val_loss_dense/np.max(val_loss_dense), label="Test Dense", c='blue', alpha=0.6)
    plt.plot(val_loss_conv /np.max(val_loss_conv ), label="Test Conv.", c='red', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer8_loss.pdf")

    plt.show()

def plot_ejercicio(history):   
     
    acc_train = 100*np.array(history.history['acc'])
    acc_test  = 100*np.array(history.history['val_acc'])

    loss = np.array(history.history['loss'])
    val_loss  = np.array(history.history['val_loss'])  
    
    return acc_train, acc_test, loss, val_loss

if __name__ == '__main__':
    ejer8()
    