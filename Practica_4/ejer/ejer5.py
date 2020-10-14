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

def logistic_training(N, epochs_total):
    training_data	=  np.ones(N).reshape(N,1)
    target_data		=  np.ones(N).reshape(N,1)

    for x in range(N):
        training_data[x][0]=np.random.uniform(0,1)
        # target_data[x][0]=logictic_function(target_data[x][0])
        for y in range(epochs_total):
        	target_data[x][0]= logictic_function(target_data[x][0])

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
    fig1.gca().set_prop_cycle(plt.cycler('color', plt.cm.gnuplot(np.linspace(0, 1, plot_num))))
    fig2.gca().set_prop_cycle(plt.cycler('color', plt.cm.gnuplot(np.linspace(0, 1, plot_num))))

    epochs_total=250
    training_data, target_data = logistic_training(115, epochs_total)

    ejemplos_array=[5,10,100]
    ejemplos_color=['red', 'black', 'blue']

    for y in range(len(ejemplos_color)):
        inp1 = layers.Input(shape=(1,))
        x = layers.Dense(5, activation='sigmoid', use_bias=True, bias_initializer='random_uniform')(inp1)
        #x = layers.Dense(main()5, activation='sigmoid')(inp1)

        x = layers.Concatenate()([inp1, x])

        output = layers.Dense(1, activation='linear', use_bias=True, bias_initializer='random_uniform')(x)
        #output = layers.Dense(1, activation='linear')(x)

        model = models.Model(inputs=inp1, outputs=output) 
        #model.summary()
        sgd = optimizers.SGD(lr=0.05,momentum=0.05)
        print(y)
        model.compile( optimizer=sgd, loss='MSE', metrics=['acc'])
        history = model.fit(training_data[:ejemplos_array[y]], target_data[:ejemplos_array[y]], validation_data=(training_data[ejemplos_array[y]:ejemplos_array[y]+14], target_data[ejemplos_array[y]:ejemplos_array[y]+14]), batch_size=14,shuffle=True, epochs=epochs_total, verbose=0)

        ax1.plot(history.history['acc'])
        ax2.plot(history.history['val_loss'], ':', label=str(ejemplos_array[y]), color=ejemplos_color[y])
        ax2.plot(history.history['loss'], '-', label=str(ejemplos_array[y]), color=ejemplos_color[y])

    ax2.legend(loc='upper right', title="Ejemplos.", ncol=3)
    ax1.legend(loc='lower right', title="Ejemplos", ncol=3)

    fig1.savefig('ejer_5_acc.pdf')
    fig2.savefig('ejer_5_los_gen.pdf')
    plt.show()


if __name__ == '__main__':
    ejer_3()
    