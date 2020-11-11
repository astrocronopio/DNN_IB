from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from pandas.tseries.offsets import DateOffset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas 

# import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.rcParams.update({
# 	'font.size': 20,
# 	'figure.figsize': [12, 8],
# 	'figure.autolayout': True,
# 	'font.family': 'serif',
# 	'font.sans-serif': ['Palatino']})
# cmap = plt.get_cmap('viridis',8)
    

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
    model.add(LSTM(100,return_sequences=True,input_shape=(l,1)))
    model.add(LSTM(100,return_sequences=True))  
    model.add(LSTM(80,return_sequences=True)) 
    model.add(LSTM(80))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.15))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=0.00005), 
                  loss='mse')
    #model.summary()
    
    return model

def Dense_model(l):
    model = Sequential()
    model.add(Dense(100,input_shape=(l,1)))
    model.add(Dense(75))  
    model.add(Dense(50)) 
    model.add(Dense(25))
    #model.add(BatchNormalization())
    model.add(Dropout(0.15))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=0.00002), 
                  loss='mse')
    #model.summary()
    
    return model

def predict_data(x, y, model, normalizar):
    predict=normalizar.inverse_transform(model.predict(x))
    mse_=np.sqrt(np.mean((y-predict)**2))  
    return predict, mse_  

def ejer(l = 1, split=0.5, plot_pass=False):
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
    model   = LSTM_model(l)
    history = model.fit(    x_train,y_train,  validation_data=(x_test,y_test),
                            epochs=100,       batch_size=1,   verbose=1)
    
    loss        = np.array(history.history['loss'])
    val_loss    = np.array(history.history['val_loss'])    
    
    train_predict, mse_train = predict_data(x_train, y_train, model, normalizar)    
    test_predict, mse_test = predict_data(x_test, y_test, model, normalizar)    

    # if plot_pass==True:
    #     plt.figure(l)
    #     plt.plot(loss/np.max(loss), c=cmap(2), label="Train")
    #     plt.plot(val_loss/np.max(val_loss), c=cmap(1), label="Test")
    #     plt.legend(loc=0)
        
    #     plot_predict(l, normalizar.inverse_transform(X_original), 
    #                     normalizar.inverse_transform(x_t), 
    #                     train_predict, test_predict)    
                
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
             label="E.", c=cmap(1))
    
    plt.plot(meses[len(train_predict):],test_predict, 
             label="V", c=cmap(0))
    
    plt.legend(loc=0)
     
if __name__ == '__main__':
    
    # ejer(2, 0.6, True)
    # plt.show()
    
    # exit()
    ls= np.arange(1,26)
    mse_train, mse_test = [] , []
    MSE_test, MSE_train = [] , []
    
    for _ in range(20):
        for l in ls: 
            print("______________________",l,"________________________")
            m_train, m_test = ejer(l, 0.7)
            mse_test.append(m_test)
            mse_train.append(m_train)
            print(mse_test)
            print(mse_train)        
        MSE_test.append(mse_test)    
        MSE_train.append(mse_train)
    
    print(MSE_test)
    print(MSE_train)    
    
    # plt.figure(33)
    # plt.plot(ls, mse_test, label="Test")
    # plt.plot(ls, mse_train, label="Train")

    # plt.show()
    
    

#[0.017683742567896843, 0.02132992073893547, 0.03018692508339882, 0.03784571588039398, 0.03718337416648865, 0.03761923313140869, 0.038386061787605286, 0.038473546504974365, 0.03537442162632942, 0.025778289884328842, 0.02434356138110161, 0.011999127455055714, 0.004787727724760771, 0.011426250450313091, 0.010823684744536877, 0.010565511882305145, 0.015495380386710167, 0.043530914932489395, 0.048162270337343216, 0.019722526893019676, 0.008757991716265678, 0.017771106213331223, 0.015204469673335552, 0.013922506012022495]
#[0.002478489186614752, 0.004578573163598776, 0.005851453170180321, 0.006586034782230854, 0.006912372540682554, 0.007209631614387035, 0.007507964037358761, 0.007678286638110876, 0.0068200696259737015, 0.006755164824426174, 0.005261139012873173, 0.002677563112229109, 0.0015345329884439707, 0.0020936299115419388, 0.002594822319224477, 0.0032060309313237667, 0.0029302032198756933, 0.0026571208145469427, 0.002395574003458023, 0.0024344264529645443, 0.0027713384479284286, 0.0027857155073434114, 0.002639304380863905, 0.0053787692449986935]
