import numpy as np 
import tensorflow as tf 
from tensorflow.keras.datasets import imdb 
from tensorflow.keras import  models, layers, optimizers, metrics 
from tensorflow.keras import  losses, activations, regularizers


def flatten(peoples_reviews, num_words):
    X = np.zeros(shape=(peoples_reviews.shape[0], num_words))
    i=0
    for review in peoples_reviews:     #Esto es MUY feo, estoy duplicando los datos
        X[i][review]+=1  #no hay gpu ni ram que aguante
        i=i+1
    return X
    

def preprocesing(num_words):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
    
    X_train = flatten(x_train, num_words)
    x_train =None
    
    X_test  = flatten(x_test, num_words)
    x_test =None
    
    mean_train = 0#X_train.mean()
    sigma_train= X_train.std()
    
    X_train = (X_train  - mean_train)/sigma_train
    X_test  = (X_test   - mean_train)/sigma_train

    return (X_train, y_train), (X_test, y_test)


def model_definition(input_shape):
    model = models.Sequential()
    
    model.add(layers.Dense(
                        units=100, 
                        input_shape=(input_shape,), 
                        use_bias=True,
                        activation=activations.relu,
                        activity_regularizer=regularizers.l2(0.010)))
    
    model.add(layers.Dropout(0.50))
    
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(
                        units=30, 
                        use_bias=True,
                        activation=activations.relu,
                        activity_regularizer=regularizers.l2(0.010)))
    
    
    model.add(layers.Dropout(0.50))
    
    model.add(layers.Dense(
                        units=1,
                        activation=activations.sigmoid,
                        activity_regularizer=regularizers.l2(0.010),
                        use_bias=True))
    
    
    model.compile(  optimizer=optimizers.SGD(0.001),
                    loss=losses.BinaryCrossentropy(), 
                    metrics=[metrics.binary_accuracy])
    
    return model
    

def ejer3():
    num_words=10000
    n_epochs=300
    (x_train, y_train), (x_test, y_test) = preprocesing(num_words)
    #print(x_test.shape)
    
    model = model_definition(x_train.shape[1])
    
    model.summary()
    #exit()
    history = model.fit(x_train, y_train, 
                        epochs=n_epochs, 
                        batch_size=50,
                        validation_data=(x_test,y_test))  

    acc = 100*np.array(history.history['binary_accuracy'])
    val_acc= 100*np.array(history.history['val_binary_accuracy'])
    
    loss = np.array(history.history['loss'])
    val_loss=  np.array(history.history['val_loss'])
    
    np.savetxt("ejer3.txt", np.array([ 
                            acc ,
                            val_acc,
                            loss ,
                            val_loss
                            ]).T)
                    

#     plot_ejercicio(history)

# def plot_ejercicio(history):
#     import matplotlib.pyplot as plt

#     import matplotlib as mpl
#     mpl.rcParams.update({
#         'font.size': 20,
#         'figure.figsize': [12, 8],
#         'figure.autolayout': True,
#         'font.family': 'serif',
#         'font.sans-serif': ['Palatino']})  
    
#     acc = 100*np.array(history.history['acc'])
#     val_acc= 100*np.array(history.history['val_acc'])
    
#     loss = np.array(history.history['loss'])
#     val_loss=  np.array(history.history['val_loss'])
          
#     plt.figure(1)
#     plt.ylabel("Precisión [%]")
#     plt.plot(acc , label="Entrenamiento", c='red', alpha=0.6, ls='--')
#     plt.plot(val_acc, label="Validación", c='blue', alpha=0.6)
#     plt.legend(loc=0)
#     plt.savefig("../docs/Figs/ejer3_acc.pdf")
    
#     plt.figure(2)
#     plt.ylabel("Pérdida")
#     plt.plot(loss    /np.max(loss    ), label="Entrenamiento", c='red', alpha=0.6, ls='--')
#     plt.plot(val_loss/np.max(val_loss), label="Validación", c='blue', alpha=0.6)
#     plt.legend(loc=0)
#     plt.savefig("../docs/Figs/ejer3_loss.pdf")

#     plt.show()
 
if __name__ == '__main__':
    ejer3()
