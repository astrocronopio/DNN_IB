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


def model_compile(loss_fun, x_shape_1):
    
    model = models.Sequential()

    model.add(layers.Dense(
                        units=100, 
                        input_shape=(x_shape_1,), 
                        use_bias=True,
                        activation=activations.sigmoid,
                        activity_regularizer=regularizers.L2(0.05)))

    model.add(layers.Dense(
                        units=10,
                        activity_regularizer=regularizers.L2(0.05),
                        use_bias=True))
    
    model.compile(  optimizer=optimizers.SGD(0.003),
                    loss=loss_fun,#losses.CategoricalCrossentropy( from_logits=True), 
                    metrics=['acc'])    
    
    return model


def ejer2_4_1():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    mean_train= x_train.mean()
    std_train=x_train.std()

    n_clasifi=10
    X, Y            = flattening(x_train,y_train, n_clasifi, mean_train, std_train)
    X_test, Y_test  = flattening(x_test ,y_test , n_clasifi, mean_train, std_train)

    
    model_1 = model_compile(losses.CategoricalCrossentropy( from_logits=True), X.shape[1])

    history_1 = model_1.fit(X, Y, 
                        epochs=500, 
                        batch_size=50,
                        validation_data=(X_test,Y_test))
    
    #acc_train_1 = np.array(history_1.history['acc'])
    acc_test_1  = np.array(history_1.history['val_acc'])
    loss_test_1  = np.array(history_1.history['val_loss'])
    
    np.savetxt("ejer2_4_1_rel.txt", np.array([ 
                            acc_test_1 ,
                            loss_test_1
                            ]).T)


def ejer2_4_2():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    mean_train= x_train.mean()
    std_train=x_train.std()

    n_clasifi=10
    X, Y            = flattening(x_train,y_train, n_clasifi, mean_train, std_train)
    X_test, Y_test  = flattening(x_test ,y_test , n_clasifi, mean_train, std_train)

    
    model_2 = model_compile(losses.MSE, X.shape[1])

    history_2 = model_2.fit(X, Y, 
                        epochs=500, 
                        batch_size=50,
                        validation_data=(X_test,Y_test))


    #acc_train_2 = np.array(history_2.history['acc'])
    acc_test_2  = np.array(history_2.history['val_acc'])
    loss_test_2  = np.array(history_2.history['val_loss'])
    
    
    np.savetxt("ejer2_4_2.txt", np.array([ 
                            acc_test_2 ,
                            loss_test_2
                            ]).T)


    
def plot_ejercicio():
    
    outputfile_mse="./ejer4_mse_v3.dat"
    outputfile_cce="./ejer4_cross_v3.dat"

    acc_test_mse, loss_test_mse = np.loadtxt(outputfile_mse, unpack=True, usecols=(1,3))
    acc_test_cce, loss_test_cce = np.loadtxt(outputfile_cce, unpack=True, usecols=(1,3))
    
    acc_test_1 , loss_test_1  = np.loadtxt("ejer2_4_1.txt", unpack=True)
    acc_test_3 , loss_test_3  = np.loadtxt("ejer2_4_1_rel.txt", unpack=True)
    acc_test_2 , loss_test_2  = np.loadtxt("ejer2_4_2.txt", unpack=True)
    
    acc_test_1=100*acc_test_1
    acc_test_2=100*acc_test_2
    acc_test_3=100*acc_test_3
    
    plt.figure(1)
    plt.xlabel("Épocas")    
    plt.ylabel("Precisión [%]")
    plt.plot(acc_test_1, label="Test - Keras - CCE", c='blue', alpha=0.6)
    plt.plot(acc_test_3, label="Test - Keras - CCE (0.05)", c='black', alpha=0.6)
    plt.plot(acc_test_2, label="Test - Keras - MSE", c='red', alpha=0.6)
    plt.plot(acc_test_cce, label="Test - CCE", c='green', alpha=0.6)
    plt.plot(acc_test_mse, label="Test - MSE", c='orange', alpha=0.6)
    
    plt.legend(loc=0)
    
    plt.figure(2)
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida Normalizada")
    plt.plot(loss_test_1  /np.max(loss_test_1  ), label="Test - Keras - CCE", c='blue', alpha=0.6)
    plt.plot(loss_test_3  /np.max(loss_test_3  ), label="Test - Keras - CCE (0.05)", c='black', alpha=0.6)
    plt.plot(loss_test_2  /np.max(loss_test_2  ), label="Test - Keras - MSE", c='red', alpha=0.6)
    plt.plot(loss_test_cce/np.max(loss_test_cce), label="Test - CCE", c='green', alpha=0.6)
    plt.plot(loss_test_mse/np.max(loss_test_mse), label="Test - MSE", c='orange', alpha=0.6)
    
    plt.legend(loc=0)


        
    
if __name__ == '__main__':
    #ejer2_4_1()
    #ejer2_4_2() 
    plot_ejercicio()
       
    plt.figure(1)
    plt.savefig("../docs/Figs/ejer2_4_acc.pdf")
    
    plt.figure(2)
    plt.savefig("../docs/Figs/ejer2_4_loss.pdf")

    plt.show()