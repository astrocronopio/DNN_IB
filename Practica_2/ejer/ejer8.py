import numpy as np
np.random.seed(54)
from keras import datasets
import modules.activation  as activation 
import modules.layer as layer
import modules.metric as metric
import modules.model as model
import modules.optimizer as optimizer
import modules.regularizador as regularizador
import modules.loss as loss
import classifier as clasificador 

import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})


def ejer8_NN1(x_train, y_train, x_test, y_test, N, NN):
    reg1 = regularizador.L1(0.001)
    reg2 = regularizador.L2(0.001)

    red_densa = model.Red()
    input_size=x_train.shape[1]

    outputfile = "ejer8.dat"

    Layer1= layer.Dense(neuronas    = x_train.shape[1], 
                        act         = activation.ReLU(), 
                        reg         = reg2,
                        name        ="Layer 1"  ,
                        bias        = True  )

    Layer2= layer.Dense(neuronas    =N, 
                        act         =activation.ReLU(), 
                        reg         =reg2,
                        name        ="Layer 2",
                        bias        = True)

    Layer3= layer.Dense(neuronas    =NN, 
                        act         =activation.Linear(), 
                        reg         =reg2,
                        name        ="Layer 3",
                        bias        = True)

    red_densa.add(Layer1)
    red_densa.add(Layer2)
    red_densa.add(Layer3)

    red_densa.fit(  
                x_train=x_train,    y_train=y_train, 
                x_test=x_test,      y_test= y_test,
                batch_size=50,
                epochs=400,
                opt=optimizer.SGD(lr=0.0002),
                loss_function=loss.cross_entropy(),
                acc_function=metric.accuracy)

    np.savetxt(outputfile,  np.array([ red_densa.acc_vect,
                            red_densa.pres_vect,
                            red_densa.loss_vect,
                            red_densa.loss_test]).T)

    # plt.figure(1)
    # plt.title("N={} - N'={} - Ejemplos {}".format(N, NN, y_train.shape[0]))
    # plt.ylabel("Accuracy [%]")
    # plt.plot(red_densa.acc_vect, label="Entrenamiento", c='red', alpha=0.6)
    # plt.plot(red_densa.pres_vect, label="Validación", c='blue', alpha=0.6)
    # plt.legend(loc=0)
    # plt.savefig("ejer8_acc.pdf")

    # plt.figure(2)
    # plt.title("N={} - N'={} - Ejemplos {}".format(N, NN, y_train.shape[0]))
    # plt.ylabel("Pérdida")
    # plt.plot(red_densa.loss_vect, label="Entrenamiento", c='red', alpha=0.6)
    # plt.plot(red_densa.loss_test, label="Validación", c='blue', alpha=0.6)
    # plt.legend(loc=0)
    # plt.savefig("ejer8_loss.pdf")
    # plt.show()


def main():
    N, NN =100, 100

    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    
    mean_train= x_train.mean()
    n_clasifi=10
    X, Y            = clasificador.flattening(x_train,y_train, n_clasifi, mean_train, False)
    X_test, Y_test  = clasificador.flattening(x_test ,y_test , n_clasifi, mean_train, False)

    print(X.shape)
    print(Y.shape)
    ejer8_NN1(X, Y, X_test, Y_test, N, NN)

if __name__ == '__main__':
    main()
    