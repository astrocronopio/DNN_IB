import numpy as np


import modules.activation  as activation 
import modules.layer as layer
import modules.metric as metric
import modules.model as model
import modules.optimizer as optimizer
import modules.regularizador as regularizador
import modules.loss as loss


import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})

# def accuracy(y_pred, y_true):
#     #y_pred = np.argmax(scores, axis=0)
#     acc = (y_pred==y_true)
#     return np.mean(acc)


def ejer7_NN1(x_train, y_train, x_test, y_test, N, NN):
    reg1 = regularizador.L1(0.001)
    reg2 = regularizador.L2(0.001)

    red_densa = model.Red()
    input_size=x_train.shape[1]

    Layer1= layer.Dense(neuronas    =N, 
                        act         =activation.sigmoid(), 
                        reg         = reg2,
                        name        ="Layer 1"  ,
                        bias        = True  )

    Layer2= layer.Dense(neuronas    =NN, 
                        act         =activation.Tanh(), 
                        reg         =reg1,
                        name        ="Layer 2",
                        bias        = True)

    red_densa.add(Layer1)
    #red_densa.add(Layer2)
    
    red_densa.fit(  
                x_train=x_train,    y_train=y_train, 
                x_test=x_test,      y_test= y_test,
                batch_size=4,
                epochs=300,
                opt=optimizer.SGD(lr=0.003),
                loss_function=loss.MSE(),
                acc_function=metric.accuracy_xor)

    plt.figure(1)
    plt.title("N={} - N'={} - Ejemplos {}".format(N, NN, y_train.shape[0]))
    plt.ylabel("Accuracy [%]")
    plt.plot(red_densa.acc_vect, label="Entrenamiento", c='red', alpha=0.6)
    plt.plot(red_densa.pres_vect, label="Validación", c='blue', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("ejer7_acc.pdf")

    plt.figure(2)
    plt.title("N={} - N'={} - Ejemplos {}".format(N, NN, y_train.shape[0]))
    plt.ylabel("Pérdida")
    plt.plot(red_densa.loss_vect, label="Entrenamiento", c='red', alpha=0.6)
    plt.plot(red_densa.loss_test, label="Validación", c='blue', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("ejer7_loss.pdf")
    plt.show()


#     input_layer = layer.Input(x_train.shape[1])
#     red_densa.add(layer.Dense(  neuronas    = 1, 
#     red_densa.add(layer.Concatenate(input_layer))
#     red_densa.add(layer.Dense(  neuronas= 1, 

def main():
    N, NN =100
    ejemplos=0
    train=4 

    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
  
    print(x_train.shape)
    print(y_train.shape)
    ejer7_NN1(x_train, y_train, x_test, y_test, N, NN)

if __name__ == '__main__':
    main()
    