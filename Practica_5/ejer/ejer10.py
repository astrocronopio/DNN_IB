import numpy as np
from tensorflow.keras.utils import to_categorical

def preprocesing(dataset):
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    # Reshaping las imagenes para que tengan un "color", ademas para hacerlos float
    x_train = x_train.reshape((x_train.shape[0], 32, 32, 3)).astype('float32')
    x_test = x_test.reshape((x_test.shape[0], 32, 32, 3)).astype('float32')
    
    mean_train = np.mean(x_train)
    x_train = (x_train - mean_train) / 255.
    x_test = (x_test - mean_train) / 255.
    
    y_train = to_categorical(y_train) #Como la guia anterior
    y_test = to_categorical(y_test)   #Para usar el categorical sin asco
    
    return (x_train, y_train), (x_test, y_test)   

def plot_ejercicio(history):   
     
    acc_train = 100*np.array(history.history['acc'])
    acc_test  = 100*np.array(history.history['val_acc'])

    loss = np.array(history.history['loss'])
    val_loss  = np.array(history.history['val_loss'])  
    
    return acc_train, acc_test, loss, val_loss



import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})

def plot_output():
    acc_vgg_10, val_acc_vgg_10, loss_vgg_10, val_loss_vgg_10 = np.loadtxt("./ejer10cifar10epochs100_vgg16.txt", unpack=True)
    acc_vgg_100, val_acc_vgg_100, loss_vgg_100, val_loss_vgg_100 = np.loadtxt("./ejer10cifar100epochs100_vgg16.txt", unpack=True)                            
    acc_alex_10, val_acc_alex_10, loss_alex_10, val_loss_alex_10 = np.loadtxt("./ejer10cifar10epochs100_alexnet.txt", unpack=True)
    acc_alex_100, val_acc_alex_100, loss_alex_100, val_loss_alex_100 = np.loadtxt("./ejer10cifar100epochs100_alexnet.txt", unpack=True)                            

    epocas = np.arange(100)
    
    plt.figure(1)
    plt.title("CIFAR-10")
    plt.ylabel("Precisión [%]")
    plt.plot(epocas, acc_vgg_10    , label="Train VGG", c='red', alpha=0.6, ls='--')
    plt.plot(epocas, val_acc_vgg_10, label="Test  VGG", c='blue', alpha=0.6)

    plt.plot(epocas, acc_alex_10    , label="Train Alex", c='orange', alpha=0.6, ls='--')
    plt.plot(epocas, val_acc_alex_10, label="Test  Alex", c='green', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer10_acc_cifar10.pdf")
    
    plt.figure(2)
    plt.title("CIFAR-10")
    plt.ylabel("Pérdida")
    plt.plot(loss_vgg_10    , label="Train VGG", c='red', alpha=0.6, ls='--')
    plt.plot(val_loss_vgg_10, label="Test  VGG", c='blue', alpha=1)

    plt.plot(loss_alex_10    , label="Train Alex", c='orange', alpha=0.6, ls='--')
    plt.plot(val_loss_alex_10, label="Test  Alex", c='green', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer10_loss_cifar10.pdf")
    
    plt.figure(3)
    plt.title("CIFAR-100")
    plt.ylabel("Precisión [%]")
    plt.plot(epocas, acc_vgg_100    , label="Train VGG", c='red', alpha=0.6, ls='--')
    plt.plot(epocas, val_acc_vgg_100, label="Test  VGG", c='blue', alpha=0.6)

    plt.plot(epocas, acc_alex_100    , label="Train Alex", c='orange', alpha=0.6, ls='--')
    plt.plot(epocas, val_acc_alex_100, label="Test  Alex", c='green', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer10_acc_cifar100.pdf")
    
    plt.figure(4)
    plt.ylabel("Pérdida")
    plt.title("CIFAR-100")
    plt.plot(loss_vgg_100    , label="Train VGG", c='red', alpha=0.6, ls='--')
    plt.plot(val_loss_vgg_100, label="Test  VGG", c='blue', alpha=0.6)

    plt.plot(loss_alex_100    , label="Train Alex", c='orange', alpha=0.6, ls='--')
    plt.plot(val_loss_alex_100, label="Test  Alex", c='green', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer10_loss_cifar100.pdf")
    print("khajb")
    plt.show()
    
if __name__ == '__main__':
    plot_output()