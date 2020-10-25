from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import losses, activations, regularizers, optimizers, metrics
import numpy as np

verbosity_mode = True
validation_split = 0.20


def padding(num_words, secuencia):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
    print(x_train.shape)
    print(x_test.shape)

    # Padding
    padded_inputs       = pad_sequences(x_train, maxlen=secuencia, value = 0) #Padding con 0 como si fuera una 
    padded_inputs_test  = pad_sequences(x_test , maxlen=secuencia, value = 0) #palabra desconocida 

    return (padded_inputs, y_train), (padded_inputs_test, y_test) 

def ejer4():
    secuencia = 400         # Voy a suponer que no existen reviews más largas que esto
    num_words = 10000       
    embedding_output = 50  #Downsizing   
    
    (x_train, y_train), (x_test, y_test) = padding(num_words, secuencia)
    print(x_train.shape)
    
    model = Sequential()
    
    model.add(Embedding(num_words, embedding_output, input_length=secuencia))
    model.add(Dropout(0.50))
    
    model.add(Conv1D(filters=16, 
                     kernel_size=4, 
                     padding='same', 
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Dropout(0.50))
        
    # model.add(Conv1D(filters=16, 
    #                  kernel_size=4, 
    #                  padding='same', 
    #                  activation='relu'))
    
    # model.add(MaxPooling1D(pool_size=2))
    
    #model.add(Dropout(0.50))
    
    model.add(Flatten())

    model.add(Dense(30, activation='relu', 
              activity_regularizer=regularizers.l2(0.01)))

    model.add(Dense(1, activation='sigmoid', 
              activity_regularizer=regularizers.l2(0.01)))

    model.compile(  optimizer=optimizers.SGD(0.002),
                    loss=losses.BinaryCrossentropy(), 
                    metrics=[metrics.binary_accuracy])

    model.summary()
    n_epochs = 300
    history = model.fit(x_train, y_train, 
                        epochs=n_epochs,
                        batch_size=50, 
                        validation_data=(x_test,y_test))
    
    
    acc, val_acc, loss, val_loss =  plot_ejercicio(history)    
    
    np.savetxt("ejer4.txt", np.array([ 
                            acc, val_acc, loss, val_loss
                            ]).T)
    
#     plt.figure(1)
#     plt.ylabel("Precisión [%]")
#     plt.plot(acc    , label="Entrenamiento", c='red', alpha=0.6, ls='--')
#     plt.plot(val_acc, label="Validación", c='blue', alpha=0.6)
#     plt.legend(loc=0)
#     plt.savefig("../docs/Figs/ejer4_acc.pdf")
    
#     plt.figure(2)
#     plt.ylabel("Pérdida")
#     plt.plot(loss    , label="Entrenamiento", c='red', alpha=0.6, ls='--')
#     plt.plot(val_loss, label="Validación", c='blue', alpha=0.6)
#     plt.legend(loc=0)
#     plt.savefig("../docs/Figs/ejer4_loss.pdf")

#     plt.show()
    
def plot_ejercicio(history):   
     
    acc_train = 100*np.array(history.history['binary_accuracy'])
    acc_test  = 100*np.array(history.history['val_binary_accuracy'])

    loss = np.array(history.history['loss'])
    val_loss  = np.array(history.history['val_loss'])  
    
    return acc_train, acc_test, loss, val_loss

import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})

def  plot_output():
    loss , acc , val_loss,  val_acc  = np.loadtxt("ejer4_saved.txt", unpack=True)
    acc3, val_acc3, loss3, val_loss3 = np.loadtxt("ejer3_optimo.txt", unpack=True)                            
    
    acc = 100*acc
    val_acc = 100*val_acc
    
    plt.figure(1)
    plt.ylabel("Precisión [%]")
    plt.plot(acc3    , label="Train Densa", c='red', alpha=0.6, ls='--')
    plt.plot(val_acc3, label="Test  Densa", c='blue', alpha=0.6)

    plt.plot(acc[100:]    , label="Train Conv.", c='orange', alpha=0.6, ls='--')
    plt.plot(val_acc[100:], label="Test  Conv.", c='green', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer4_acc.pdf")
    
    plt.figure(2)
    plt.ylabel("Pérdida")
    plt.plot(loss3    , label="Train Conv.", c='red', alpha=0.6, ls='--')
    plt.plot(val_loss3, label="Test  Conv.", c='blue', alpha=1)

    plt.plot(loss    , label="Train Conv.", c='orange', alpha=0.6, ls='--')
    plt.plot(val_loss, label="Test  Conv.", c='green', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer4_loss.pdf")

    plt.show()
if __name__ == '__main__':
    ejer4()
    #plot_output()