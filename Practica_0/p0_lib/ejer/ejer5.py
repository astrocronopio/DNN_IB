""" Versión 4: Antes de agregar la clase loss y activation
    Versión 5: Separé las activaciones y losses en otro modulo
               También agregué la activación de la segunda capa.
    Versión 6: Separé la clase classifier al modulo classifier.py    
"""
""""
ejer5.py
    Versión 1: Le agregué el cambio en la activación de la capa final
"""

import numpy as np
from tensorflow.keras import datasets
#np.random.seed(648)
#import matplotlib.pyplot as plt
import modules.activation as act
import modules.loss as los
import classifier as clasificador 
import modules.regularizador as regularizador

# import matplotlib as mpl
# mpl.rcParams.update({
# 	'font.size': 20,
# 	'figure.figsize': [12, 8],
# 	'figure.autolayout': True,
# 	'font.family': 'serif',
# 	'font.sans-serif': ['Palatino']})


def ejer5_loss(loss_fun, act_fun_last, label, nfig1, nfig2):
    print(label)
    reg1 = regularizador.L2(0.1)
    reg2 = regularizador.L2(0.1)

    proto= clasificador.Classifier(
                    epochs    =300,
                    batch_size=50,
                    eta       =0.001)

    outputfile="ejer5_"+label+"_v3.dat"

    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    
    mean_train= x_train.mean()

    n_clasifi=10
    X, Y            = clasificador.flattening(x_train,y_train, n_clasifi, mean_train)
    X_test, Y_test  = clasificador.flattening(x_test ,y_test , n_clasifi, mean_train)

    proto.fit(
            X, Y, X_test, Y_test,
            act_function1 = act.ReLU(0),
            reg1=reg1,
            loss_function= loss_fun,
            act_function2= act_fun_last,
            reg2=reg2)

    # plt.figure(nfig1)
    # plt.ylabel("Accuracy [%]")
    # plt.plot(proto.acc_vect, label="Entrenamiento", c='red', alpha=0.6, ls='--')
    # plt.plot(proto.pres_vect, label="Validación", c='blue', alpha=0.6)
    # plt.legend(loc=0)
    # plt.savefig("ejer5_acc_"+label+".pdf")
    

    # plt.figure(nfig2)
    # plt.ylabel("Pérdida")
    # plt.plot(proto.loss_vect, label="Entrenamiento", c='red', alpha=0.6, ls='--')
    # plt.plot(proto.loss_test, label="Validación", c='blue', alpha=0.6)
    # plt.legend(loc=0)
    # plt.savefig("ejer5_loss_"+label+".pdf")
    # plt.show()
    #plt.close()
    #plt.clf()
    
    
    np.savetxt(outputfile,   np.array([
                             proto.acc_vect ,
                             proto.pres_vect,
                             proto.loss_vect,
                             proto.loss_test]).T)

def ejer5():

    #ejer5_loss(los.cross_entropy(), act.sigmoid(), "cross-sig-v4", 1,2)
    
    # LLegó a 51 % en v4 y lr 0.0007#
    #ejer5_loss(los.cross_entropy(), act.ReLU_Linear(), "cross-ReLuLineal-v4", 3,4)
    
    ejer5_loss(los.MSE(), act.sigmoid(), "mse-sig_v5", 5,6)
    
    # LLegó a  52% en vv4 con 0.0005
    #ejer5_loss(los.MSE(), act.ReLU_Linear(), "mse-ReLuLineal-v4", 7,8)
    pass

def main():
    ejer5()
    pass

if __name__ == '__main__':
    main()
    
