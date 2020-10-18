import numpy as np 
import tensorflow as tf 
from tensorflow.keras.datasets import imdb 
from tensorflow.keras import  models, layers, optimizers 
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
    del x_train
    
    X_test  = flatten(x_test, num_words)
    x_test =None
    del x_test
    
    mean_train = 0#X_train.mean()
    sigma_train= X_train.std()
    
    X_train = (X_train  - mean_train)/sigma_train
    X_test  = (X_test   - mean_train)/sigma_train

    return (X_train, y_train), (X_test, y_test)


def model_definition(input_shape):
    model = models.Sequential()
    
    model.add(layers.Dense(
                        units=10, 
                        input_shape=(input_shape,), 
                        use_bias=True,
                        activation=activations.relu,
                        activity_regularizer=regularizers.L2(0.000)))
    
    model.add(layers.Dropout(0.50))
    
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(
                        units=5, 
                        use_bias=True,
                        activation=activations.sigmoid,
                        activity_regularizer=regularizers.L2(0.000)))
    
    
    model.add(layers.Dropout(0.50))
    
    model.add(layers.Dense(
                        units=1,
                        activity_regularizer=regularizers.L2(0.000),
                        use_bias=True))
    
    
    model.compile(  optimizer=optimizers.SGD(0.003),
                    loss=losses.BinaryCrossentropy(), 
                    metrics=['acc'])
    
    return model
    

def ejer3():
    num_words=20
    n_epochs=2
    (x_train, y_train), (x_test, y_test) = preprocesing(num_words)
    print(x_test.shape)
    
    model = model_definition(x_train.shape[1])
    
    model.summary()
    #exit()
    history = model.fit(x_train, y_train, 
                        epochs=n_epochs, 
                        batch_size=20,
                        validation_data=(x_test,y_test))  

    plot_ejercicio(history)

def plot_ejercicio(history):
    import matplotlib.pyplot as plt

    import matplotlib as mpl
    mpl.rcParams.update({
        'font.size': 20,
        'figure.figsize': [12, 8],
        'figure.autolayout': True,
        'font.family': 'serif',
        'font.sans-serif': ['Palatino']})  
          
    plt.figure(1)
    plt.ylabel("Precisión [%]")
    plt.plot(100*history.history['acc']    , label="Entrenamiento", c='red', alpha=0.6, ls='--')
    plt.plot(100*history.history['val_acc'], label="Validación", c='blue', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer3_acc.pdf")
    
    plt.figure(2)
    plt.ylabel("Pérdida")
    plt.plot(history.history['loss']    , label="Entrenamiento", c='red', alpha=0.6, ls='--')
    plt.plot(history.history['val_loss'], label="Validación", c='blue', alpha=0.6)
    plt.legend(loc=0)
    plt.savefig("../docs/Figs/ejer3_loss.pdf")

    plt.show()
    
if __name__ == '__main__':
    ejer3()