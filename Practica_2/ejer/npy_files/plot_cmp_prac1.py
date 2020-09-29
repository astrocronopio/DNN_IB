import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})


def main():
    #Orden con el que salve los pra1**.py
    with open('pra1_ejer5_MNIST.npy', 'rb') as f:
        SVM_m_error_loss =  np.load(f)#, allow_pickle=True)
        # SMC_m_error_loss =  np.load(f)#, allow_pickle=True)
        # SVM_m_loss_test =   np.load(f)#, allow_pickle=True)
        # SMC_m_loss_test =   np.load(f)#, allow_pickle=True)
        # SVM_m_error_acc =   np.load(f)#, allow_pickle=True)
        # SMC_m_error_acc =   np.load(f)#, allow_pickle=True)
        # SVM_m_error_pres =  np.load(f)#, allow_pickle=True)
        # SMC_m_error_pres =  np.load(f)#, allow_pickle=True)
    
    print(SVM_m_error_loss)
    print("ep")
    plt.plot(SVM_m_error_loss)
    plt.show()


if __name__ == "__main__":
    main()
    pass