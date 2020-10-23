import numpy as np 
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras import losses, activations, regularizers
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})

cmap = plt.get_cmap('nipy_spectral',5)

def flattening(x, y, n_clasifi, mean_train, std_train=1 ):
    X= np.copy(x) 
    X= np.reshape(X, (X.shape[0], np.prod(x.shape[1:])))
    X= (X - mean_train)/std_train
    
    Y = np.zeros(shape=(y.shape[0], n_clasifi))
    y_aux  = np.copy(y).reshape(y.shape[0])

    Y[np.arange(Y.shape[0]), y_aux]=1
    
    #y_train = to_categorical(y_train)
    #y_test  = to_categorical(y_test)
    
    return X,Y


def ejer2_3():
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
                        use_bias=True)
            )
    model.compile(  optimizer=optimizers.SGD(0.003),
                    loss=losses.MSE, 
                    metrics=['acc'])

    n_epochs=400
    history = model.fit(X, Y, 
                        epochs=n_epochs, 
                        batch_size=50,
                        validation_data=(X_test,Y_test))


    acc_train = 100*np.array(history.history['acc'])
    acc_test  = 100*np.array(history.history['val_acc'])
    
    loss_train =  np.array(history.history['loss'])
    loss_test_keras  = np.array(history.history['val_loss'])
    
    
    outputfile='ejer3_v2_mse.dat'

    acc_vect, pres_vect, loss_vect, loss_test = np.loadtxt(outputfile, unpack=True)

    
    plt.figure(1)
    plt.xlabel("Épocas")
    epocas = np.arange(n_epochs)
    
    plt.ylabel("Precisión [%]")
    plt.plot(epocas, acc_train  , label="Train - Keras ", c='red', alpha=0.6, ls='--')
    plt.plot(epocas, acc_test, label="Test - Keras", c='blue', alpha=0.6)
    
    plt.plot(epocas, acc_vect[:n_epochs], label="Train", c='green', alpha=0.6, ls='--')
    plt.plot(epocas, pres_vect[:n_epochs], label="Test", c='orange', alpha=0.6)
    
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer2_3_acc.pdf")
    
    plt.figure(2)
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.plot(epocas, loss_train[:n_epochs]/np.max(loss_train[:n_epochs])   , label="Train - Keras", c='red', alpha=0.6, ls='--')
    plt.plot(epocas, loss_test_keras[:n_epochs]/np.max(loss_test_keras[:n_epochs]), label="Test - Keras", c='blue', alpha=0.6)
    
    plt.plot(epocas, loss_vect[:n_epochs]/np.max(loss_vect[:n_epochs]), label="Train", c='green', alpha=0.6, ls='--')
    plt.plot(epocas, loss_test[:n_epochs]/np.max(loss_test[:n_epochs]), label="Test",c='orange', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer2_3_loss.pdf")

    plt.show()
    
if __name__ == '__main__':
    ejer2_3()