from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

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
    
def model_definition(num_classes):
    
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['acc'])
    return model

def ejer8():
    (x_train, y_train), (x_test, y_test) = preprocesing()

    num_classes = y_test.shape[1]

    model = model_definition(num_classes)

    history = model.fit(x_train, y_train, 
                        epochs=10, 
                        batch_size=100,
                        validation_data=(x_test, y_test))
    
    plt.figure(1)
    plt.ylabel("Precisión [%]")
    plt.plot(100*history.history['acc']    , label="Entrenamiento", c='red', alpha=0.6, ls='--')
    plt.plot(100*history.history['val_acc'], label="Validación", c='blue', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer8_acc.pdf")
    
    plt.figure(2)
    plt.ylabel("Pérdida")
    plt.plot(history.history['loss']    , label="Entrenamiento", c='red', alpha=0.6, ls='--')
    plt.plot(history.history['val_loss'], label="Validación", c='blue', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer8_loss.pdf")
    plt.show()


if __name__ == '__main__':
    ejer8()
    