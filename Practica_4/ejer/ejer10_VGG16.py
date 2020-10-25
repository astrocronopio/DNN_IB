from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras import losses, activations, regularizers, optimizers, metrics
import numpy as np
from tensorflow.keras.utils import to_categorical

from ejer10  import preprocesing, plot_ejercicio

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt

# import matplotlib as mpl
# mpl.rcParams.update({
# 	'font.size': 20,
# 	'figure.figsize': [12, 8],
# 	'figure.autolayout': True,
# 	'font.family': 'serif',
# 	'font.sans-serif': ['Palatino']})


def model_vgg16_cifar(n_clasif, xshape):
    input_shape = xshape[1:]
    
    model = Sequential()
    # 2 x Conv
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # 2 x Conv
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    #model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # 3 x Conv 
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    #model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # 3 x Conv
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    #model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    #model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # 3 x Conv
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    #model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    #model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(1000, activation='relu', activity_regularizer=regularizers.l2(0.001)))
    model.add(Dense(1000 , activation='relu', activity_regularizer=regularizers.l2(0.001)))
    #model.add(Dense(4096 , activation='relu'))
    model.add(Dense(n_clasif , activation='linear',  activity_regularizer=regularizers.l2(0.001) ))


    model.summary()

    # Compile the model
    model.compile(loss=losses.categorical_crossentropy(from_logits=True), 
                  optimizer=optimizers.Adam(learning_rate=0.001), #optimizers.SGD(lr=0.03), 
                  metrics=[metrics.CategoricalAccuracy('acc')]) 
    return model

def ejer10_vgg16(dataset,  label, n_epochs, n_clasif):
    (x_train, y_train), (x_test, y_test) = preprocesing(dataset)
    
    model =  model_vgg16_cifar(n_clasif, x_train.shape)
    
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
    
    np.savetxt("ejer10{}epochs{}_vgg16.txt".format(label, n_epochs),
                np.array([ 
                            acc, val_acc, loss, val_loss
                            ]).T)

if __name__ == '__main__':
    n_epochs=100
    #ejer10_vgg16(cifar10, "cifar10", n_epochs, 10)
    ejer10_vgg16(cifar100, "cifar100", n_epochs, 100)
