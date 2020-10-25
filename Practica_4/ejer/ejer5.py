import numpy as np 
import tensorflow as tf 
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras import losses, activations, regularizers


import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
'font.size': 20,
'figure.figsize': [12, 8],
'figure.autolayout': True,
'font.family': 'serif',
'font.sans-serif': ['Palatino']})


def logictic_function(x):
    return 4*x*(1-x)

def logistic_training(N):
    training_data=np.random.uniform(0,1, size=(N,1))
    target_data = logictic_function(training_data)                                      

    return training_data, target_data

def ejer_3_1():
    #fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    # ax1.set_title('Exactitud')
    # ax1.set_xlabel('Épocas')
    # ax1.set_ylabel('Exactitud')
    ax2.set_title(' Entrenamiento (-), Validación ($\\cdot\\cdot$)')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Pérdida Normalizada')

    plot_num=6

    epochs_total=200
    ej_vec=  np.array([10,24,50], dtype=np.int)
    training_data, target_data = logistic_training(120)

    ejemplos_color=['red', 'black', 'blue', 'orange']

    for y in range(len(ej_vec)):
        
        inp1 = layers.Input(shape=(1,))
        x = layers.Dense(5, activation='tanh', use_bias=True, bias_initializer='random_uniform')(inp1)
        x = layers.Concatenate()([inp1, x])
        
        output = layers.Dense(1, activation='linear', 
                                use_bias=True, 
                                bias_initializer='random_uniform')(x)
       
        model = models.Model(inputs=inp1, outputs=output) 
        #model.summary()
        sgd = optimizers.SGD(lr=0.008)
        #print(y)ss
        model.compile( optimizer=sgd, loss='MSE', metrics=['mse'])
        
        size_vec=ej_vec[y]
        test_size= int(size_vec*0.5)
        
        x_train=training_data[:size_vec]
        y_train=target_data[:size_vec]
        
        x_test= training_data[size_vec:size_vec+test_size]
        y_test=   target_data[size_vec:size_vec+test_size]
        
        history = model.fit(x_train, y_train,
                            validation_data=(x_test , y_test ), 
                            epochs=epochs_total)

        loss = np.array(history.history['loss'])
        val_loss = np.array(history.history['val_loss'])
        
        epocas = np.arange(len(loss))
        
        #ax1.plot(epocas, acc, label=str(size_vec), color=ejemplos_color[y])
        ax2.plot(epocas, val_loss/np.max(val_loss), ':', label=str(test_size), alpha=0.6,  color=ejemplos_color[y])
        ax2.plot(epocas, loss/np.max(loss),         '-', label=str(size_vec) , alpha=0.6,  color=ejemplos_color[y])
        tf.keras.backend.clear_session()
        
    ax2.legend(loc=0, title="Ejemplos", ncol=3)
    #ax1.legend(loc='lower right', title="Ejemplos", ncol=3)

    #fig1.savefig('../docs/Figs/ejer_5_acc.pdf')
    fig2.savefig('../docs/Figs/ejer_5_los_gen.pdf')
    plt.show()


if __name__ == '__main__':
    ejer_3_1()
    