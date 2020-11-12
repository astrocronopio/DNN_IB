from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from pandas.tseries.offsets import DateOffset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas 

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})
cmap = plt.get_cmap('viridis',8)
    

#por que es necesario? Porque el scaler usa la
# primera columna en adelante si es numpy arrays
def item_1(): 
    x_t = pandas.read_csv("./airline-passengers.csv", header='infer')
    x_t["Month"] = pandas.to_datetime(x_t["Month"])
    x_t.set_index("Month", inplace=True)
    x_t.index.name = "Fecha"
    return  x_t

def format_data(data, l=1):
    X_original, X, Y = [], [] ,[]
    # -l por el Y, -1 para no pisar memoria en X
    for i in range(len(data)-l-1): 
        dummy = data[i:(i+l), 0]   #[x(t), x(t+1)..., x(t+l-1)]
        X.append(dummy)
        Y.append(data[i + l, 0]) # x(t+l)
    X_original = np.array(data[0:,])    
    X,Y = np.array(X), np.array(Y)    
    return X_original, np.array(X), np.array(Y)

def split_data(x_t, y_t, split):
    split_data = int(split*len(x_t))
    
    train_x_t, test_x_t = x_t[:split_data], x_t[split_data:]
    train_y_t, test_y_t = y_t[:split_data], y_t[split_data:]
    return train_x_t,  train_y_t, test_x_t, test_y_t

def scale_data(x_t):
    normalizar = MinMaxScaler()
    normalizar.fit(x_t)
    x_t = normalizar.transform(x_t)
    return x_t, normalizar

def add_noise(x_t):
    ruido = np.random.normal(0,0.02, x_t.shape)
    x_t += ruido 
    return x_t

def LSTM_model(l):
    model = Sequential()
    model.add(LSTM(4,input_shape=(l,1)))
    # model.add(LSTM(100,return_sequences=True))  
    # model.add(LSTM(80,return_sequences=True)) 
    # model.add(LSTM(80))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.15))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=0.002), 
                  loss='mse')
    #model.summary()
    
    return model

def Dense_model(l):
    model = Sequential()
    model.add(Dense(4,input_shape=(l,1)))
    # model.add(Dense(75))  
    # model.add(Dense(50)) 
    # model.add(Dense(25))
    # #model.add(BatchNormalization())
    # model.add(Dropout(0.15))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=0.002), 
                  loss='mse')
    #model.summary()
    
    return model

def predict_data(x, y, model, normalizar):
    y_pred = model.predict(x)
    y_pred=y_pred.reshape((y_pred.shape[0], 1))
    print(y_pred.shape)
    predict=normalizar.inverse_transform(y_pred)
    mse_=np.sqrt(np.mean((y-predict)**2))  
    return predict, mse_  

def ejer(l = 1, split=0.5, plot_pass=False, custom_model=LSTM_model):
    x_t = item_1()

    #item 2
    x_t, normalizar  = scale_data(x_t)  
    X_original, x_t, y_t = format_data(x_t,l)
    
    #item 3
    x_t = add_noise(x_t)     

    #item 4
    x_train, y_train, x_test, y_test= split_data(x_t, y_t, split)  
    
    #item 5
    x_train=x_train.reshape(x_train.shape[0], l, 1)
    x_test=x_test.reshape(x_test.shape[0], l, 1)

    #item 6 e item 7
    model   = custom_model(l)
    history = model.fit(    x_train,y_train,  validation_data=(x_test,y_test),
                            epochs=100,       batch_size=1,   verbose=0)
    
    loss        = np.array(history.history['loss'])
    val_loss    = np.array(history.history['val_loss'])    
    
    train_predict, mse_train = predict_data(x_train, y_train, model, normalizar)    
    test_predict, mse_test = predict_data(x_test, y_test, model, normalizar)    

    if plot_pass==True:
        plt.figure(l)
        plt.xlabel("Épocas")
        plt.ylabel("Pérdida Normalizada")
        plt.plot(loss/np.max(loss), c=cmap(2), label="Train")
        plt.plot(val_loss/np.max(val_loss), c='red', label="Test")
        plt.legend(loc=0)
        
        plot_predict(l, normalizar.inverse_transform(X_original), 
                        normalizar.inverse_transform(x_t), 
                        train_predict, test_predict)    
                
    return loss[-1], val_loss[-1]

def plot_predict(l,X_original ,x_t, train_predict, test_predict):   
    plt.figure(20+l) 

    plt.xlabel("Mes")
    plt.ylabel("Pasajeros")
    meses = np.arange(len(x_t))

    plt.plot(X_original, label="Datos", 
             c=cmap(5),alpha=0.4, ls=':')
    
    plt.plot(x_t[:,0], label="Datos + Ruido", 
             c=cmap(2), alpha=0.4, ls='--')
    
    plt.plot(meses[:len(train_predict)],train_predict, 
             label="E.", c='red', alpha=0.6)
    
    plt.plot(meses[len(train_predict):],test_predict, 
             label="V", c=cmap(0), alpha=0.6)
    
    plt.legend(loc=0)
     
if __name__ == '__main__':
    
    print(ejer(1, 0.7, True))
    plt.show()
    
    exit()
    
    ls= np.arange(1,15)
    mse_train= [] 
    mse_test = [] 

    MSE_test =  []
    MSE_train = []
    

    for _ in range(1):
        MSE_test =  []
        MSE_train = []
        for l in ls: 
            print("______________",l,"_________________")
            m_train, m_test = ejer(l, 0.5)
            mse_test.append(m_test)
            mse_train.append(m_train)
                    
        MSE_test.append(mse_test)    
        MSE_train.append(mse_train)
    
    print(MSE_test)
    print(MSE_train)   

    # with open('./drive/My Drive/file.txt', 'w') as f:
    #     print(MSE_test, file=f)
    #     print(MSE_train, file=f)   
    
    MSE_test= np.array(MSE_test)
    MSE_train= np.array(MSE_train)
    
    A = np.mean(MSE_test, axis=0)
    B = np.mean(MSE_train, axis=0)
    
    plt.figure(33)
    plt.ylabel("MSE")
    plt.xlabel("l")
    plt.plot(ls, A, c='red' , alpha=0.6,  label="Test")
    #plt.plot(ls, B, c='blue', alpha=0.6,  label="Train")

    plt.show()
