import numpy as np 
from tensorflow.keras.datasets import cifar10
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


def flattening(x, y, n_clasifi, mean_train, std_train=1 ):
    X= np.copy(x) 
    X= np.reshape(X, (X.shape[0], np.prod(x.shape[1:])))
    X= (X - mean_train)/std_train
    
    Y = np.zeros(shape=(y.shape[0], n_clasifi))
    y_aux  = np.copy(y).reshape(y.shape[0])

    Y[np.arange(Y.shape[0]), y_aux]=1
    return X,Y


def ejer2_4():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    mean_train= x_train.mean()
    std_train=x_train.std()

    n_clasifi=10
    X, Y            = flattening(x_train,y_train, n_clasifi, mean_train, std_train)
    X_test, Y_test  = flattening(x_test ,y_test , n_clasifi, mean_train, std_train)

    
    model = models.Sequential()

    model.add(layers.Dense(
                        units=100, 
                        input_shape=(X.shape[1],), 
                        use_bias=True,
                        activation=activations.sigmoid,
                        activity_regularizer=regularizers.L2(0.0001)))

    model.add(layers.Dense(
                        units=10,
                        activity_regularizer=regularizers.L2(0.0001),
                        use_bias=True))
    
    model.compile(  optimizer=optimizers.SGD(0.003),
                    loss=losses.CategoricalCrossentropy( from_logits=True), 
                    metrics=['acc'])

    history = model.fit(X, Y, 
                        epochs=400, 
                        batch_size=50,
                        validation_data=(X_test,Y_test))


    acc_train = history.history['acc']
    acc_test  = history.history['val_acc']
    
    
    plt.figure(1)
    plt.xlabel("Épocas")    
    plt.ylabel("Precisión [%]")
    plt.plot(history.history['acc']    , label="Entrenamiento", c='red', alpha=0.6, ls='--')
    plt.plot(history.history['val_acc'], label="Validación", c='blue', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("ejer2_4_acc.pdf")
    
    plt.figure(2)
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.plot(history.history['loss']    , label="Entrenamiento", c='red', alpha=0.6, ls='--')
    plt.plot(history.history['val_loss'], label="Validación", c='blue', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer2_4_loss.pdf")

    plt.close()
    
if __name__ == '__main__':
    ejer2_4()