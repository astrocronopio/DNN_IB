import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 19,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Computer Modern Roman']})


cmap = plt.get_cmap('nipy_spectral',5)


#Ejer3
def plot_ejer3():
    reference_practica_1="./ejer5_CIFAR-10_v2.dat"
    outputfile='ejer3_v2_mse.dat'

    acc_vect, pres_vect, loss_vect, loss_test = np.loadtxt(outputfile, unpack=True)
    SVM_pres, SMC_pres = np.loadtxt(reference_practica_1, unpack=True, usecols=(6,7))

    plt.figure(1)
    plt.ylabel("Precisión [%]")
    plt.plot(acc_vect, label="Train: {:.4}%".format(acc_vect[-1]), c=cmap(0), alpha=0.6, ls='--')
    plt.plot(pres_vect, label="Test: {:.4}%".format(pres_vect[-1]), c=cmap(1))
    plt.legend(loc=4)
    plt.savefig("ejer3_acc.pdf")

    plt.figure(2)
    plt.ylabel("Pérdida Normalizada")
    plt.plot(loss_vect/np.max(loss_vect), label="Train", c=cmap(0), alpha=0.6, ls='--')
    plt.plot(loss_test/np.max(loss_test), label="Test",c=cmap(1))
    plt.legend(loc=0)
    plt.savefig("ejer3_loss.pdf")

    plt.figure(3)
    plt.ylabel("Precisión [%]")
    plt.plot(pres_vect[:400], label="lr=0.003: {:.4}%".format(pres_vect[-1]), c=cmap(0))
    plt.plot(SVM_pres, label="SVM, lr=0.0002: {:.4}%".format(SVM_pres[-1]), c=cmap(1))
    plt.plot(SMC_pres, label="SMC, lr=0.0002: {:.4}%".format(SMC_pres[-1]), c=cmap(2))

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
    plt.ylabel("Precisión [%]")
    plt.plot(pres_vect_cce  , label="Test CCE: {:.4}%".format(pres_vect_cce[-1]), c=cmap(0))
    plt.plot(acc_vect_cce   , label="Train CCE: {:.4}%".format(acc_vect_cce [-1]),alpha=0.6,  c=cmap(1),  ls='--')
    plt.plot(pres_vect_mse  , label="Test MSE: {:.4}%".format(pres_vect_mse[-1]), c=cmap(2))  
    plt.plot(acc_vect_mse   , label="Train MSE: {:.4}%".format(acc_vect_mse [-1]),alpha=0.6,  c=cmap(3), ls='--')
    plt.legend(loc=4)
    plt.savefig("ejer4_acc.pdf")

    plt.figure(5)
    plt.ylabel("Pérdida Normalizada")
    plt.plot(loss_vect_cce/np.max(loss_vect_cce),c=cmap(1), label="Train CCE",alpha=0.6,  ls='--')
    plt.plot(loss_test_cce/np.max(loss_test_cce),c=cmap(0), label="Test CCE")
    plt.plot(loss_vect_mse/np.max(loss_vect_mse),c=cmap(3), label="Train MSE",alpha=0.6,  ls='--')
    plt.plot(loss_test_mse/np.max(loss_test_mse),c=cmap(2), label="Test MSE")
    plt.legend(loc=0)
    plt.savefig("ejer4_loss.pdf")

    plt.figure(6)
    plt.ylabel("Precisión [%]")
    plt.plot(pres_vect_cce[:400], label="CCE, lr=0.003: {:.4}%".format(pres_vect_cce[-1]), c=cmap(1))
    plt.plot(pres_vect_mse[:400], label="MSE, lr=0.003: {:.4}%".format(pres_vect_mse[-1]), c=cmap(0))
    plt.plot(SVM_pres, label="SVM, lr=0.0002: {:.4}%".format(SVM_pres[-1]), c=cmap(3))
    plt.plot(SMC_pres, label="SMC, lr=0.0002: {:.4}%".format(SMC_pres[-1]), c=cmap(2))

    plt.legend(loc=0)
    plt.savefig("ejer4_acc_all.pdf")

def plot_ejer5():
    cmap = plt.get_cmap('nipy_spectral',10)

    out_cross_sig="./ejer5_cross-sig_v4.dat"
    out_mse_sig="./ejer5_mse-sig_v5.dat"

    ####
    out_cross_ReLU="./ejer5_cross-ReLuLineal-v4_v3.dat"    
    out_mse_ReLU="./ejer5_mse-ReLuLineal-v4_v3.dat"

    reference_practica_1="./ejer5_CIFAR-10_v2.dat"
    SVM_pres, SMC_pres = np.loadtxt(reference_practica_1, unpack=True, usecols=(6,7))

    acc_vect_cross_sig , pres_vect_cross_sig , loss_vect_cross_sig , loss_test_cross_sig  = np.loadtxt(out_cross_sig , unpack=True)
    acc_vect_cross_ReLU, pres_vect_cross_ReLU, loss_vect_cross_ReLU, loss_test_cross_ReLU = np.loadtxt(out_cross_ReLU, unpack=True)
    acc_vect_mse_sig   , pres_vect_mse_sig   , loss_vect_mse_sig   , loss_test_mse_sig    = np.loadtxt(out_mse_sig   , unpack=True)
    acc_vect_mse_ReLU  , pres_vect_mse_ReLU  , loss_vect_mse_ReLU  , loss_test_mse_ReLU   = np.loadtxt(out_mse_ReLU  , unpack=True)

    plt.figure(11)
    plt.ylabel("Pérdida Normalizada")
    plt.ylim((0.4,1.2))
    plt.xlim(left=-10)
    plt.xlim(right=300)
    plt.plot(loss_vect_cross_ReLU[:]/np.max(loss_vect_cross_ReLU[:]), label="Train R+L -CCE", c=cmap(1), alpha=0.6, ls='--')
    plt.plot(loss_test_cross_ReLU[:]/np.max(loss_test_cross_ReLU[:]), label="Test R+L -CCE", c=cmap(1))
    plt.plot(loss_vect_cross_sig[:]/np.max(loss_vect_cross_sig[:]), label="Train S -CCE", c=cmap(2), alpha=0.6, ls='--')
    plt.plot(loss_test_cross_sig[:]/np.max(loss_test_cross_sig[:]), label="Test S -CCE", c=cmap(2))
    plt.plot(loss_vect_mse_ReLU[:]/np.max(loss_vect_mse_ReLU), label="Train R+L -MSE", c=cmap(3), alpha=0.6, ls='--')
    plt.plot(loss_test_mse_ReLU[:]/np.max(loss_test_mse_ReLU), label="Test R+L -MSE", c=cmap(3))
    plt.plot(loss_vect_mse_sig[:]/np.max(loss_vect_mse_sig), label="Train S -MSE", c=cmap(4), alpha=0.6, ls='--')
    plt.plot(loss_test_mse_sig[:]/np.max(loss_test_mse_sig), label="Test S -MSE", c=cmap(4))
    plt.legend(loc=0, ncol=2)
    #plt.show()
    plt.savefig("ejer5_acc.pdf")


    plt.figure(12)
    plt.ylabel("Precisión [%]")
    plt.xlim(right=300)
    plt.xlim(left=-10)
    plt.plot(acc_vect_cross_ReLU[:], label="Train R+L -CCE: {:.4}%".format(acc_vect_cross_ReLU[-1]), c=cmap(0), alpha=0.6, ls='--')
    plt.plot(pres_vect_cross_ReLU[:], label="Test R+L -CCE: {:.4}%".format(pres_vect_cross_ReLU[-1]), c=cmap(1))
    plt.plot(acc_vect_cross_sig[:], label="Train Sig-CCE: {:.4}%".format(acc_vect_cross_sig[-1]), c=cmap(2), alpha=0.6, ls='--')
    plt.plot(pres_vect_cross_sig[:], label="Test Sig-CCE: {:.4}%".format(pres_vect_cross_sig[-1]), c=cmap(3))
    plt.plot(acc_vect_mse_ReLU[:], label="Train R+L -MSE: {:.4}%".format(acc_vect_mse_ReLU[-1]), c=cmap(4), alpha=0.6, ls='--')
    plt.plot(pres_vect_mse_ReLU[:], label="Test R+L -MSE: {:.4}%".format(pres_vect_mse_ReLU[-1]), c=cmap(5))
    plt.plot(acc_vect_mse_sig[:], label="Train S-MSE: {:.4}%".format(acc_vect_mse_sig[-1]), c=cmap(6), alpha=0.6, ls='--')
    plt.plot(pres_vect_mse_sig[:], label="Test S-MSE: {:.4}%".format(pres_vect_mse_sig[-1]), c=cmap(7))
    plt.legend(loc=0, ncol=2)
    plt.savefig("ejer5_loss.pdf")

    plt.figure(13)
    plt.xlim(right=300)
    plt.xlim(left=-10)
    plt.ylabel("Precisión [%]")
    plt.plot(pres_vect_cross_ReLU[:], label="Test R+L -CCE: {:.4}%".format(pres_vect_cross_ReLU[-1]), c=cmap(1))
    plt.plot(pres_vect_cross_sig[:], label="Test Sig-CCE: {:.4}%".format(pres_vect_cross_sig[-1]), c=cmap(2))
    plt.plot(pres_vect_mse_ReLU[:], label="Test R+L -MSE: {:.4}%".format(pres_vect_mse_ReLU[-1]), c=cmap(3))
    plt.plot(pres_vect_mse_sig[:], label="Test S-MSE: {:.4}%".format(pres_vect_mse_sig[-1]), c=cmap(4))
    plt.plot(SVM_pres, label="SVM, lr=0.0002: {:.4}%".format(SVM_pres[-1]), c=cmap(5))
    plt.plot(SMC_pres, label="SMC, lr=0.0002: {:.4}%".format(SMC_pres[-1]), c=cmap(6))
    plt.legend(loc=0, ncol=2)
    plt.savefig("ejer5_acc_all.pdf")


def plot_ejer8():
    cmap = plt.get_cmap('gnuplot',6)
    reference_practica_1="./ejer5_CIFAR-10_v2.dat"
    outputfile="./ejer8.dat"
    outputfile_sigmoidal="./ejer8_v3_sigmoid.dat"
    outputfile3="./ejer3_v2_mse.dat"

    acc_vect, pres_vect, loss_vect, loss_test = np.loadtxt(outputfile, unpack=True)
    acc_vect_sigmoidal, pres_vect_sigmoidal, loss_vect_sigmoidal, loss_test_sigmoidal = np.loadtxt(outputfile_sigmoidal, unpack=True)
    acc_vect3, pres_vect3, loss_vect3, loss_test3 = np.loadtxt(outputfile3, unpack=True)
    
    SVM_pres, SMC_pres = np.loadtxt(reference_practica_1, unpack=True, usecols=(6,7))

    plt.figure(7)
    plt.ylabel("Precisión [%]")
    plt.plot(acc_vect[:], label="Train ReLU: {:.4}%".format(acc_vect[-1]), c=cmap(0), alpha=0.6, ls='--')
    plt.plot(pres_vect[:], label="Test ReLU: {:.4}%".format(pres_vect[-1]), c=cmap(1))
    plt.plot(acc_vect_sigmoidal[:], label="Train Sig: {:.4}%".format(acc_vect_sigmoidal[-1]), c=cmap(2), alpha=0.6, ls='--')
    plt.plot(pres_vect_sigmoidal[:], label="Test Sig: {:.4}%".format(pres_vect_sigmoidal[-1]), c=cmap(3))

    plt.legend(loc=4)
    plt.savefig("ejer8_acc.pdf")

    plt.figure(8)
    plt.ylabel("Pérdida Normalizada")
    plt.plot(loss_vect[:]/np.max(loss_vect), label="Train ReLU", c=cmap(0), alpha=0.6, ls='--')
    plt.plot(loss_test[:]/np.max(loss_test), label="Test ReLU",c=cmap(1))
    plt.plot(loss_vect_sigmoidal[:]/np.max(loss_vect_sigmoidal), label="Train Sig", c=cmap(2), alpha=0.6, ls='--')
    plt.plot(loss_test_sigmoidal[:]/np.max(loss_test_sigmoidal), label="Test Sig",c=cmap(3))

    plt.legend(loc=0)
    plt.savefig("ejer8_loss.pdf")

    plt.figure(9)
    plt.ylabel("Precisión [%]")
    plt.plot(pres_vect[:], label="ReLU, lr=0.001: {:.4}%".format(pres_vect[-1]), c=cmap(0))
    plt.plot(pres_vect_sigmoidal[:], label="Sig, lr=0.001: {:.4}%".format(pres_vect_sigmoidal[-1]), c=cmap(1))
    plt.plot(SVM_pres, label="SVM, lr=0.0002: {:.4}%".format(SVM_pres[-1]), c=cmap(3))
    plt.plot(SMC_pres, label="SMC, lr=0.0002: {:.4}%".format(SMC_pres[-1]), c=cmap(2))
    plt.plot(pres_vect3[:400], label="Ejer3, lr=0.003: {:.4}%".format(pres_vect3[-1]), c=cmap(4))

    plt.legend(loc=0)
    plt.savefig("ejer8_acc_all.pdf")

def main():
    #plot_ejer3() # DONE
    #plot_ejer4() # DONE
    plot_ejer5() 
    #plot_ejer8() # DONE


    #plt.close()
    plt.show()

if __name__ == '__main__':
    main()
    