from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras import losses, activations, regularizers, optimizers, metrics
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})


def model_alex_net_cifar():
    model = Sequential()
    
    input_shape =(227,227,3)

    # 1 Conv
    model.add(Conv2D(       filters=96          ,input_shape=input_shape,
                            kernel_size=(11,11) ,strides=(4,4),
                            padding='same'      ,activation='relu'))
    model.add(MaxPooling2D( pool_size=(2,2),strides=(2,2)          , padding='same'))

    # 2 Conv
    model.add(Conv2D(filters=256,   kernel_size=(5,5), strides=(1,1), 
                     padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    # 3 Conv
    model.add(Conv2D(filters=384,   kernel_size=(3,3), strides=(1,1), 
                    padding='same', activation='relu'))

    # 4 Conv
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), 
                    padding='same', activation='relu'))

    # 5 Conv 
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), 
                    padding='same',activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    # Fully Connected
    model.add(Flatten())
    
    # 1 FC
    model.add(Dense(4096, input_shape=(227*227*3,), activation='relu'))
    model.add(Dropout(0.4))

    # 2 FC
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))

    # 3 FC
    model.add(Dense(1000, activation='softmax'))
    model.add(Dropout(0.4))

    # # Output Layer
    # model.add(Dense(10, activation='softmax'))
    # model.summary()

    # Compile the model
    model.compile(loss=losses.categorical_crossentropy, 
                optimizer='adam', 
                metrics=['acc']) 

def ejer10_alexnet(dataset, n_epochs=100):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    model = model_alex_net_cifar()
    model.fit(  x_train, y_train, 
                epochs=n_epochs, 
                validation_data=(x_test,y_test))


if __name__ == '__main__':
    ejer10_alexnet(cifar10);

