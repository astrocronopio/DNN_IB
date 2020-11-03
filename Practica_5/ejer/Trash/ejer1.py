from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras import losses, activations, regularizers, optimizers, metrics
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


from imageio import  imread
from os import listdir

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from random import shuffle

 
def files_loading(factor):
    filepath= "/home/ponci/Desktop/Datos/dogs-vs-cats_small/"
    files = listdir(filepath)
    
    x=np.array([])
    y=np.array([])
    
    shuffle(files)
    files = files[:2000]  
    #print(files)  
    
    for image_path in files:
        path = filepath+image_path
        iscat = 1 if image_path.find("cat")!=-1 else 0
        y = np.append(y, iscat)
        
        image = tf.keras.preprocessing.image.load_img(path) #, target_size=(299, 299, 3))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        x= np.append(x, input_arr)  
                
    x = x.reshape((len(files), 3072)) 
    y = y.reshape((len(files), 1))

    fin = int(len(files)*factor)
    x_train, y_train, x_test, y_test = x[fin:], y[fin:] ,x[:fin], y[:fin,]
    
    return (x_train, y_train), (x_test, y_test)


def preprocesing():
    (x_train, y_train), (x_test, y_test) = files_loading(0.25)
    # Reshaping las imagenes para que tengan un "color", ademas para hacerlos float
    
    x_train = x_train.reshape((x_train.shape[0], 32, 32, 3)).astype('float32')
    x_test = x_test.reshape((x_test.shape[0], 32, 32, 3)).astype('float32')
    
    mean_train = np.mean(x_train)
    sig_train = np.std(x_train)
    x_train = (x_train - mean_train) / sig_train
    x_test = (x_test - mean_train) / sig_train
    
    #y_train = to_categorical(y_train) #Como la guia anterior
    #y_test = to_categorical(y_test)   #Para usar el categorical sin asco
    
    return (x_train, y_train), (x_test, y_test)   

def plot_ejercicio(history):   
     
    acc_train = 100*np.array(history.history['acc'])
    acc_test  = 100*np.array(history.history['val_acc'])

    loss = np.array(history.history['loss'])
    val_loss  = np.array(history.history['val_loss'])  
    
    return acc_train, acc_test, loss, val_loss

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
    model.add(Dense(500, activation='relu', activity_regularizer=regularizers.l2(0.001)))
    model.add(Dense(500 , activation='relu', activity_regularizer=regularizers.l2(0.001)))
    #model.add(Dense(4096 , activation='relu'))
    model.add(Dense(n_clasif , activation='linear',  activity_regularizer=regularizers.l2(0.001) ))

    model.summary()

    # Compile the model
    model.compile(loss=losses.BinaryCrossentropy(from_logits=True), 
                  optimizer=optimizers.Adam(learning_rate=0.0001), #optimizers.SGD(lr=0.03), 
                  metrics=[metrics.BinaryAccuracy('acc')]) 
    return model

def ejer1_vgg16(n_epochs, n_clasif):
    (x_train, y_train), (x_test, y_test) = preprocesing()
    print(x_train.shape)
    
    model =  model_vgg16_cifar(n_clasif, x_train.shape)
    
    IDG = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=5,
    height_shift_range=5,
    shear_range=0.0,  
    fill_mode='nearest',  
    horizontal_flip=True,  
    vertical_flip=True)  
    
    # history = model.fit(IDG.flow(x_train, y_train, batch_size=100),    
    #             epochs=n_epochs, 
    #             validation_data=(x_test,y_test))
    
    history = model.fit(x_train, y_train, batch_size=10,    
                epochs=n_epochs, 
                validation_data=(x_test,y_test))
        
    
    acc, val_acc, loss, val_loss =  plot_ejercicio(history)    
    
    np.savetxt("ejer1{}epochs{}_vgg16.txt".format("small_4000", n_epochs),
                np.array([ acc, val_acc, loss, val_loss ]).T)

if __name__ == '__main__':
    n_epochs=50
    ejer1_vgg16(n_epochs, 1)
    
    #(x_train, y_train), (x_test, y_test) = preprocesing()
    
    # print(x_test.shape)
    # print(x_train.shape)