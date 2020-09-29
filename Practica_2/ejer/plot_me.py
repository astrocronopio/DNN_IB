import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Computer Modern Roman']})


cmap = plt.get_cmap('nipy_spectral',4)


#Ejer3
def plot_ejer3():
    reference_practica_1="./ejer5_CIFAR-10_v2.dat"
    outputfile='ejer3_v2_mse.dat'

    acc_vect, pres_vect, loss_vect, loss_test = np.loadtxt(outputfile, unpack=True)
    SVM_pres, SMC_pres = np.loadtxt(reference_practica_1, unpack=True, usecols=(6,7))

    plt.figure(1)
    plt.ylabel("Accuracy [%]")
    plt.plot(acc_vect, label="Entrenamiento: {}".format(acc_vect[-1]), c=cmap(0), alpha=0.6, ls='--')
    plt.plot(pres_vect, label="Validación: {}".format(acc_vect[-1]), c=cmap(1))
    plt.legend(loc=4)
    plt.savefig("ejer3_acc.pdf")

    plt.figure(2)
    plt.ylabel("Pérdida")
    plt.plot(loss_vect/np.max(loss_vect), label="Entrenamiento", c=cmap(0), alpha=0.6, ls='--')
    plt.plot(loss_test/np.max(loss_test), label="Validación",c=cmap(1))
    plt.legend(loc=0)
    plt.savefig("ejer3_loss.pdf")

    plt.figure(3)
    plt.ylabel("Accuracy [%]")
    plt.plot(pres_vect[:400], label="lr=0.003 ->{}".format(pres_vect[-1]), c=cmap(0))
    plt.plot(SVM_pres, label="SVM, lr=0.0002 ->".format(SVM_pres[-1]), c=cmap(1))
    plt.plot(SMC_pres, label="SMC, lr=0.0002 ->".format(SMC_pres[-1]), c=cmap(2))

    plt.legend(loc=0)
    plt.savefig("ejer3_acc_all.pdf")


    

def plot_ejer4():
    reference_practica_1="./ejer5_CIFAR-10_v2.dat"
    outputfile_mse="./ejer4_mse_v3.dat"
    outputfile_cce="./ejer4_cross_v3.dat"

    acc_vect_mse, pres_vect_mse, loss_vect_mse, loss_test_mse = np.loadtxt(outputfile_mse, unpack=True)
    acc_vect_cce, pres_vect_cce, loss_vect_cce, loss_test_cce = np.loadtxt(outputfile_cce, unpack=True)
    SVM_pres, SMC_pres = np.loadtxt(reference_practica_1, unpack=True, usecols=(6,7))

    plt.figure(4)
    plt.ylabel("Accuracy [%]")
    plt.plot(pres_vect_cce  , label="Test CCE: {}".format(pres_vect_cce[-1]), c=cmap(0))
    plt.plot(acc_vect_cce   , label="Train CCE: {}".format(acc_vect_cce [-1]),alpha=0.6,  c=cmap(1),  ls='--')
    plt.plot(pres_vect_mse  , label="Test MSE: {}".format(pres_vect_mse[-1]), c=cmap(2))  
    plt.plot(acc_vect_mse   , label="Train MSE: {}".format(acc_vect_mse [-1]),alpha=0.6,  c=cmap(3), ls='--')
    plt.legend(loc=4)
    plt.savefig("ejer4_acc.pdf")

    plt.figure(5)
    plt.ylabel("Pérdida")
    plt.plot(loss_vect_cce/np.max(loss_vect_cce),c=cmap(1), label="Train CCE",alpha=0.6,  ls='--')
    plt.plot(loss_test_cce/np.max(loss_test_cce),c=cmap(0), label="Test CCE")
    plt.plot(loss_vect_mse/np.max(loss_vect_mse),c=cmap(3), label="Train MSE",alpha=0.6,  ls='--')
    plt.plot(loss_test_mse/np.max(loss_test_mse),c=cmap(2), label="Test MSE")
    plt.legend(loc=0)
    plt.savefig("ejer4_loss.pdf")

    plt.figure(6)
    plt.ylabel("Accuracy [%]")
    plt.plot(pres_vect_cce[:400], label="CCE, lr=0.003", c=cmap(1))
    plt.plot(pres_vect_mse[:400], label="MSE, lr=0.003", c=cmap(0))
    plt.plot(SVM_pres, label="SVM, lr=0.0002", c=cmap(3))
    plt.plot(SMC_pres, label="SMC, lr=0.0002", c=cmap(2))

    plt.legend(loc=0)
    plt.savefig("ejer4_acc_all.pdf")

def plot_ejer8():
    reference_practica_1="./ejer5_CIFAR-10_v2.dat"
    outputfile="./ejer8.dat"

    acc_vect, pres_vect, loss_vect, loss_test = np.loadtxt(outputfile, unpack=True)
    SVM_pres, SMC_pres = np.loadtxt(reference_practica_1, unpack=True, usecols=(6,7))

    plt.figure(7)
    plt.ylabel("Accuracy [%]")
    plt.plot(acc_vect[:], label="Entrenamiento", c=cmap(0), alpha=0.6, ls='--')
    plt.plot(pres_vect[:], label="Validación", c=cmap(1))
    plt.legend(loc=4)
    plt.savefig("ejer8_acc.pdf")

    plt.figure(8)
    plt.ylabel("Pérdida")
    plt.plot(loss_vect[:]/np.max(loss_vect), label="Entrenamiento", c=cmap(0), alpha=0.6, ls='--')
    plt.plot(loss_test[:]/np.max(loss_test), label="Validación",c=cmap(1))
    plt.legend(loc=0)
    plt.savefig("ejer8_loss.pdf")

    plt.figure(9)
    plt.ylabel("Accuracy [%]")
    plt.plot(pres_vect[:], label="lr=0.001", c=cmap(0))
    plt.plot(SVM_pres[:], label="SVM, lr=0.0002", c=cmap(1))
    plt.plot(SMC_pres[:], label="SMC, lr=0.0002", c=cmap(2))

    plt.legend(loc=0)
    plt.savefig("ejer8_acc_all.pdf")


        

def main():
    plot_ejer3()
    plot_ejer4()
    #plot_ejer5()
    plot_ejer8()



    plt.show()

if __name__ == '__main__':
    main()
    