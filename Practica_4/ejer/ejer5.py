import numpy as np 

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

def ejer_3():
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    ax1.set_title('Exactitud')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Exactitud')
    ax2.set_title(' Entrenamiento (-), Generalización ($\\cdot\\cdot$)')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Error')

    plot_num=6

    epochs_total=250
    x_test = 15
    ej_vec=[5,25,50]
    training_data, target_data = logistic_training(100 + x_test)

    ejemplos_color=['red', 'black', 'blue']

    for y in range(len(ej_vec)):
        
        inp1 = layers.Input(shape=(1,))
        x = layers.Dense(5, activation='sigmoid', use_bias=True, bias_initializer='random_uniform')(inp1)
        #x = layers.Dense(main()5, activation='sigmoid')(inp1)
        x = layers.Concatenate()([inp1, x])
        output = layers.Dense(1, activation='linear', 
                                use_bias=True, 
                                bias_initializer='random_uniform')(x)
        #output = layers.Dense(1, activation='linear')(x)

        model = models.Model(inputs=inp1, outputs=output) 
        #model.summary()
        sgd = optimizers.SGD(lr=0.05)
        #print(y)
        model.compile( optimizer=sgd, loss='MSE', metrics=['mse'])
        
        history = model.fit(training_data[:ej_vec[y]], 
                            target_data[:ej_vec[y]], 
                            validation_data=(training_data[ej_vec[y]:ej_vec[y]+x_test], 
                                             target_data[ej_vec[y]:ej_vec[y]+x_test]), 
                            epochs=epochs_total)

        acc = np.array(history.history['mse'])
        epocas = np.arange(len(acc))
        
        ax1.plot(epocas, acc, label=str(ej_vec[y]), color=ejemplos_color[y])
        ax2.plot(history.history['val_loss'], ':', label=str(ej_vec[y]), color=ejemplos_color[y])
        ax2.plot(history.history['loss'], '-', label=str(ej_vec[y]), color=ejemplos_color[y])

    ax2.legend(loc='upper right', title="Ejemplos.", ncol=3)
    ax1.legend(loc='lower right', title="Ejemplos", ncol=3)

    fig1.savefig('../docs/Figs/ejer_5_acc.pdf')
    fig2.savefig('../docs/Figs/ejer_5_los_gen.pdf')
    plt.show()


if __name__ == '__main__':
    ejer_3()
    