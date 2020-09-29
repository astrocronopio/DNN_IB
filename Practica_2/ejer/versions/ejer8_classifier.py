import numpy as np

np.random.seed(54)
import matplotlib.pyplot as plt
import modules.activation as act
import modules.loss as los
import classifier as clasificador 

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


def ejer8_221(x_train, y_train):
    reg1 = regularizador.L1(0.0)
    reg2 = regularizador.L2(0.0)

    red_densa = model.Red()
    input_size=x_train.shape[1]

    Layer1= layer.Dense(neuronas    =2, 
                        act         =activation.Tanh(), 
                        reg         = reg2,
                        name        ="Layer 1"  ,
                        bias        = True  )

    Layer2= layer.Dense(neuronas    =2, 
                        act         =activation.Tanh(), 
                        reg         =reg1,
                        name        ="Layer 2",
                        bias        = True)

    red_densa.add(Layer1)
    red_densa.add(Layer2)
    
    red_densa.fit(  
                x_train=x_train,    y_train=y_train, 
                batch_size=4,
                epochs=100,
                opt=optimizer.SGD(lr=0.1),
                loss_function=loss.MSE(),
                acc_function=metric.accuracy_xor)

    plt.figure(1)
    plt.ylabel("Accuracy [%]")
    plt.plot(red_densa.acc_vect, label="Entrenamiento", c='red', alpha=0.6)
    #plt.plot(red_densa.pres_vect, label="Validación", c='blue', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("ejer8_acc.pdf")

    plt.figure(2)
    plt.ylabel("Pérdida")
    plt.plot(red_densa.loss_vect, label="Entrenamiento", c='red', alpha=0.6)
    #plt.plot(red_densa.loss_test, label="Validación", c='blue', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("ejer8_loss.pdf")
    plt.show()


#     input_layer = layer.Input(x_train.shape[1])
#     red_densa.add(layer.Dense(  neuronas    = 1, 
#     red_densa.add(layer.Concatenate(input_layer))
#     red_densa.add(layer.Dense(  neuronas= 1, 

def main():
    x_train = np.array([[-1, -1], [-1, 1], [-1, 1], [1,1]])
    y_train = np.array([1], [-1], [-1], [1]])

    ejer8_221(x_train, y_train)
    #ejer8_221(x_train, y_train)

if __name__ == '__main__':
    main()
    