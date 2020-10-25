import numpy as np
import itertools 
#np.random.seed(55479)
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

cmap = plt.get_cmap('gist_rainbow',6)
# def accuracy(y_pred, y_true):
#     #y_pred = np.argmax(scores, axis=0)
#     acc = (y_pred==y_true)
#     return np.mean(acc)


def ejer7_NN1(x_train, y_train, x_test, y_test, N, NN, ejemplos, i):
    reg1 = regularizador.L1(0.0)
    reg2 = regularizador.L2(0.0)

    red_densa = model.Red()
    input_size=x_train.shape[1]

    Layer1= layer.Dense(neuronas    =input_size, 
                        act         =activation.Tanh(), 
                        reg         = reg2,
                        name        ="Layer 1"  ,
                        bias        = True  )

    Layer2= layer.Dense(neuronas    =NN, 
                        act         =activation.Tanh(), 
                        reg         =reg1,
                        name        ="Layer 2",
                        bias        = True)

    red_densa.add(Layer1)
    red_densa.add(Layer2)
    
    red_densa.fit(  
                x_train=x_train,    y_train=y_train, 
                x_test=x_test,      y_test= y_test,
                batch_size= ejemplos,
                epochs=500,
                opt=optimizer.SGD(lr=0.1),
                loss_function=loss.MSE(),
                acc_function=metric.accuracy_xor)
    plt.figure(1)

    plt.ylabel("Accuracy [%]")
    plt.plot(red_densa.acc_vect, c=cmap(i), label="({},{},{})".format(N, NN, ejemplos))
    plt.legend(loc=0)


    plt.figure(2)
    plt.ylabel("Pérdida Normalizada")
    plt.plot(red_densa.loss_vect/np.max(red_densa.loss_vect), c=cmap(i),label="({},{},{})".format(N, NN, ejemplos))
    plt.legend(loc=0)
    

def create_data(N, ejemplos=0):
    train=2**N - ejemplos
    X=np.array(list(itertools.product([-1.0,1.0], repeat=N)), dtype=float)
    Y=np.array([[np.prod(p)] for p in X], dtype=float)

    x_test = None if ejemplos==0 else X[train: train + ejemplos]
    y_test = None if ejemplos==0 else Y[train: train + ejemplos]
    
    return X[:train],  Y[:train] ,x_test, y_test 

def main():
    plt.title("(N,N',Ejemplos)")
    i=0

    N, NN = 7, 7
    x_train, y_train , x_test, y_test = create_data(N)
    ejer7_NN1(x_train, y_train, x_test, y_test, N, 2, 15, i)
    i=i+1
    ejer7_NN1(x_train, y_train, x_test, y_test, N, 7, 15, i)
    i=i+1
    ejer7_NN1(x_train, y_train, x_test, y_test, N, 12, 15, i)
    i=i+1

    N, NN = 10, 10
    x_train, y_train , x_test, y_test = create_data(N)
    ejer7_NN1(x_train, y_train, x_test, y_test, N, 3, 20, i)
    i=i+1
    ejer7_NN1(x_train, y_train, x_test, y_test, N, 25, 20, i)

    #plt.savefig("ejer7_acc.pdf")
    plt.show()
if __name__ == '__main__':
    main()
    