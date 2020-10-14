from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})



n_epochs = 100
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
    secuencia = 100
    num_words = 10000 #100 x 100
    embedding_output = 15    
    
    (x_train, y_train), (x_test, y_test) = padding(num_words, secuencia)
    print(x_train.shape)
    
    model = Sequential()
    model.add(Embedding(num_words, embedding_output, input_length=secuencia))
    model.add(Dropout(0.50))
    model.add(Conv1D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(Dropout(0.50))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dropout(0.50))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['acc'])

    model.summary()

    history = model.fit(x_train, y_train, 
                        epochs=n_epochs, 
                        validation_data=(x_test,y_test))
        
    
    plt.figure(1)
    plt.ylabel("Precisión [%]")
    plt.plot(100*history.history['acc']    , label="Entrenamiento", c='red', alpha=0.6, ls='--')
    plt.plot(100*history.history['val_acc'], label="Validación", c='blue', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer4_acc.pdf")
    
    plt.figure(2)
    plt.ylabel("Pérdida")
    plt.plot(history.history['loss']    , label="Entrenamiento", c='red', alpha=0.6, ls='--')
    plt.plot(history.history['val_loss'], label="Validación", c='blue', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer4_loss.pdf")

    plt.close()
    
if __name__ == '__main__':
    ejer4()