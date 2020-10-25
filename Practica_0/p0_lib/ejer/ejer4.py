""" Versión 4: Antes de agregar la clase loss y activation
    Versión 5: Separé las activaciones y losses en otro modulo
               También agregué la activación de la segunda capa.
    Versión 6: Separé la clase classifier al modulo classifier.py    
"""

import numpy as np
from tensorflow.keras import datasets
np.random.seed(54)

import modules.activation as act
import modules.loss as los
import classifier as clasificador 
import modules.regularizador as regularizador

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})


def ejer4_loss(loss_fun, label, nfig1, nfig2):
    print(label)
    
    reg1 = regularizador.L2(0.0001)
    reg2 = regularizador.L2(0.0001)
    proto= clasificador.Classifier(
                    epochs    =400,
                    batch_size=50,
                    eta       =0.003)
                      
    outputfile="ejer4_"+label+"_v3.dat"

    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    
    mean_train= x_train.mean()

    n_clasifi=10
    X, Y            = clasificador.flattening(x_train,y_train, n_clasifi, mean_train)
    X_test, Y_test  = clasificador.flattening(x_test ,y_test , n_clasifi, mean_train)

    proto.fit(
            X, Y, X_test, Y_test,
            act_function1 = act.sigmoid(),
            reg1=reg1,
            loss_function= loss_fun,
            act_function2= act.Linear(),
            reg2=reg2)

    plt.figure(nfig1)
    plt.ylabel("Accuracy [%]")
    plt.plot(proto.acc_vect,  label="Entrenamiento", c='red', alpha=0.6, ls='--')
    plt.plot(proto.pres_vect, label="Validación", c='blue', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("ejer4_acc_"+label+".pdf")
    

    plt.figure(nfig2)
    plt.ylabel("Pérdida")
    plt.plot(proto.loss_vect, label="Entrenamiento", c='red', alpha=0.6, ls='--')
    plt.plot(proto.loss_test, label="Validación", c='blue', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("ejer4_loss_"+label+".pdf")
    plt.close()

    np.savetxt(outputfile,  np.array([
                         proto.acc_vect,
                         proto.pres_vect,
                         proto.loss_vect,
                         proto.loss_test]).T)

def ejer4():
    ejer4_loss(los.cross_entropy(), "cross", 1, 2)
    ejer4_loss(los.MSE(), "mse", 3, 4 )

def main():
    ejer4()
    pass

if __name__ == '__main__':
    main()
    