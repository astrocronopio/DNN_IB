from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras import losses, activations, regularizers, optimizers, metrics
import numpy as np
from tensorflow.keras.utils import to_categorical


from ejer10  import preprocesing, plot_ejercicio

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# import matplotlib.pyplot as plt

# import matplotlib as mpl
# mpl.rcParams.update({
#   'font.size': 20,
#   'figure.figsize': [12, 8],
#   'figure.autolayout': True,
#   'font.family': 'serif',
#   'font.sans-serif': ['Palatino']})


def model_alex_net_cifar(n_clasif, xshape):
    model = Sequential()
    
    input_shape =xshape
    print(xshape[1:])

    # 1 Conv
    model.add(Conv2D(       filters=96          ,input_shape=input_shape[1:],
                            kernel_size=(11,11) ,strides=   (2,2),
                            padding='same'      ,activation='relu'))
    
    model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2), padding='same'))

    # 2 Conv
    model.add(Conv2D(filters=256,   kernel_size=(5,5), strides=(1,1), 
                     padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    # # 3 Conv
    # model.add(Conv2D(filters=384,   kernel_size=(3,3), strides=(1,1), 
    #                 padding='same', activation='relu'))

    # 4 Conv
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), 
                    padding='same', activation='relu'))

    # 5 Conv 
    model.add(Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), 
                    padding='same',activation='relu'))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))

    # Fully Connected
    model.add(Flatten())
    model.add(BatchNormalization())
    # 1 FC
    # model.add(Dense(2048, activation='relu'))
    # model.add(Dropout(0.4))

    # 2 FC
    model.add(Dense(2048, activation='relu'))

    # 3 FC
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(n_clasif, activation='softmax'))
    model.summary()

    # Compile the model
    model.compile(loss=losses.categorical_crossentropy, 
                optimizer=optimizers.Adam(learning_rate=0.001), 
                metrics=[metrics.CategoricalAccuracy('acc')]) 
    
    return model
    

def ejer10_alexnet(dataset, label,n_epochs, n_clasif):
    (x_train, y_train), (x_test, y_test) =preprocesing(dataset)
    
    model = model_alex_net_cifar(n_clasif, x_train.shape)
    
    IDG = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=5,
    height_shift_range=5,
    shear_range=0.1,  
    fill_mode='nearest',  
    horizontal_flip=True,  
    vertical_flip=True)  
    
    history = model.fit(IDG.flow(x_train, y_train, batch_size=100),
                        epochs=n_epochs, 
                        validation_data=(x_test,y_test))
        
    acc, val_acc, loss, val_loss =  plot_ejercicio(history)    
    
    np.savetxt("ejer10{}epochs{}_alexnet.txt".format(label, n_epochs),
                np.array([ 
                            acc, val_acc, loss, val_loss
                            ]).T)

if __name__ == '__main__':
    n_epochs=100
    ejer10_alexnet(cifar10,"cifar10", n_epochs, 10)
    #ejer10_alexnet(cifar100,"cifar100", n_epochs, 100)

