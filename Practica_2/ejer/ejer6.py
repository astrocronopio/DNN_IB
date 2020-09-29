import numpy as np

#np.random.seed(548)

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

cmap = plt.get_cmap('nipy_spectral',4)

def ejer6_221(x_train, y_train):
    reg1 = regularizador.L2(0.0)
    reg2 = regularizador.L2(0.0)

    red_densa = model.Red()
    input_size=x_train.shape[1]

    Layer1= layer.Dense(neuronas    = input_size, 
                        act         = activation.Tanh(), 
                        reg         = reg1,
                        name        ="Layer 1"  ,
                        bias        = True  )

    Layer2= layer.Dense(neuronas    = 2, 
                        act         = activation.Tanh(), 
                        reg         = reg2,
                        name        = "Layer 2",
                        bias        = True)

    red_densa.add(Layer1)
    red_densa.add(Layer2)
    
    red_densa.fit(  
                x_train=x_train,    y_train=y_train, 
                batch_size=4,
                epochs=200,
                opt=optimizer.SGD(lr=0.01),
                loss_function=loss.MSE(),
                acc_function=metric.accuracy_xor)

    plt.figure(1)
    plt.ylabel("Accuracy [%]")
    plt.plot(red_densa.acc_vect, label="Entrenamiento", c=cmap(0), alpha=0.6)
    #plt.plot(red_densa.pres_vect, label="Validación", c='blue', alpha=0.6)
    plt.legend(loc=0)
    #plt.savefig("ejer6_acc.pdf")

    plt.figure(2)
    plt.ylabel("Pérdida")
    plt.plot(red_densa.loss_vect, label="Entrenamiento", c=cmap(2), alpha=0.6)
    #plt.plot(red_densa.loss_test, label="Validación", c='blue', alpha=0.6)
    plt.legend(loc=0)
    #plt.savefig("ejer6_loss.pdf")
    plt.show()


def ejer6_211(x_train, y_train):
    reg1 = regularizador.L1(0.0)
    reg2 = regularizador.L2(0.0)

    red_densa = model.Red()
    input_size=x_train.shape[1]

    Layer1= layer.Dense(neuronas    =input_size, 
                        act         =activation.Tanh(), 
                        reg         = reg2,
                        name        ="Layer 1"  ,
                        bias        = True  )

    Layer2= layer.Dense(neuronas    =1, 
                        act         =activation.Tanh(), 
                        reg         =reg1,
                        name        ="Layer 2",
                        bias        = True)

    red_densa.add(Layer1)

    layer_aux= layer.ConcatInput()
    
    red_densa.add(layer_aux(input_size, Layer2))

    #red_densa.add(Layer2)
    
    red_densa.fit(  
                x_train=x_train,    y_train=y_train, 
                batch_size=4,
                epochs=300,
                opt=optimizer.SGD(lr=0.2),
                loss_function=loss.MSE(),
                acc_function=metric.accuracy_xor)

    plt.figure(1)
    plt.ylabel("Accuracy [%]")
    plt.plot(red_densa.acc_vect, label="Entrenamiento", c='red', alpha=0.6)
    plt.legend(loc=0)
    #plt.savefig("ejer6_acc_211.pdf")

    plt.figure(2)
    plt.ylabel("Pérdida")
    plt.plot(red_densa.loss_vect, label="Entrenamiento", c='red', alpha=0.6)
    plt.legend(loc=0)
    #plt.savefig("ejer6_loss_211.pdf")
    plt.show()

def main():
    x_train = np.array([[-1.0, -1.0],  [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0],])
    y_train = np.array([   [1.0],       [-1.0],      [-1.0],      [1.0]])

    ejer6_221(x_train, y_train)
    #ejer6_211(x_train, y_train)

if __name__ == '__main__':
    main()
    