""" Versión 4: Antes de agregar la clase loss y activation
    Versión 5: Separé las activaciones y losses en otro modulo
               También agregué la activación de la segunda capa.
    Versión 6: Separé la clase classifier al modulo classifier.py    
"""

import numpy as np
from keras import datasets
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


def ejer3():
    proto= clasificador.Classifier(
                    epochs    =200,
                    batch_size=50,
                    eta       =0.003,
                    lambda_L2 =0.0001)
    
    outputfile='ejer3.npy'

    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    
    mean_train= x_train.mean()

    n_clasifi=10
    X, Y            = clasificador.flattening(x_train,y_train, n_clasifi, mean_train)
    X_test, Y_test  = clasificador.flattening(x_test ,y_test , n_clasifi, mean_train)

    proto.fit(
            X, Y, X_test, Y_test,
            act_function = act.sigmoid(),
            loss_function= los.MSE(),
            act_function2= act.Linear())

    plt.figure(1)
    plt.ylabel("Accuracy [%]")
    plt.plot(proto.acc_vect, label="Entrenamiento", c='red', alpha=0.6, ls='--')
    plt.plot(proto.pres_vect, label="Validación", c='blue', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("ejer3_acc.pdf")
    
    np.save(outputfile, proto.acc_vect )
    np.save(outputfile, proto.pres_vect )

    plt.figure(2)
    plt.ylabel("Pérdida")
    plt.plot(proto.loss_vect, label="Entrenamiento", c='red', alpha=0.6, ls='--')
    plt.plot(proto.loss_test, label="Validación", c='blue', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("ejer3_loss.pdf")

    np.save(outputfile, proto.loss_vect)
    np.save(outputfile, proto.loss_test)

    plt.show()
    pass

def main():
    ejer3()
    pass

if __name__ == '__main__':
    main()
    