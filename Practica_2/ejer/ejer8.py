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



def ejer8_221():
    x_train = np.array([[-1, -1], [-1, 1], [-1, 1], [1,1]])
    y_train = np.array([[1], [-1], [-1], [1]])

    


    reg1 = regularizador.L1(0.1)
    reg2 = regularizador.L2(0.1)

    red_densa = model.Red()

    red_densa.add(layer.Dense(  neuronas=2, 
                            activation=activation.Tanh(), 
                            input_size=x_train.shape[1],
                            reg=reg1))

    red_densa.add(layer.Dense(  neuronas      =1, 
                            activation=activation.Tanh(), 
                            reg=reg1))
    
    red_densa.fit(  
                x_train=x_train,    y_train=y_train, 
                batch_size=4,
                epochs=200,
                opt=optimizer.SGD(lr=0.05),
                loss=loss.MSE())
    pass

def ejer8_211():
    x_train = np.array([[-1, -1], [-1, 1], [-1, 1], [1,1]])
    y_train = np.array([[1], [-1], [-1], [1]])


    reg1 = regularizador.L1(0.1)
    reg1 = regularizador.L1(0.1)


    red_densa = model.Red()

    input_layer = layer.Input(x_train.shape[1])

    red_densa.add(layer.Dense(  unit      =1, 
                            activation=activation.Tanh(), 
                            input_size=x_train.shape[1]),
                            reg=reg1)

    red_densa.add(layer.Concatenate(input_layer))

    red_densa.add(layer.Dense(  unit      =1, 
                            activation=activation.Tanh(), 
                            reg=reg2))


    red_densa.fit(  x_train=x_train,    y_train=y_train, 
                batch_size=4,
                epochs=200,
                opt=optimizer.SGD(lr=0.05),
                loss=loss.MSE())
    pass


def main():
    ejer8_221()
    ejer8_221()

if __name__ == '__main__':
    main()
    