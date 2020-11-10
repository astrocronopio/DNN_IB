from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from pandas.tseries.offsets import DateOffset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

import numpy as np
import pandas 
import matplotlib.pyplot as plt

#por que es necesario? Porque el scaler usa la
# primera columna en adelante si es numpy arrays
def item_1(): 
    x_t = pandas.read_csv("./airline-passengers.csv", header='infer')
    x_t["Month"] = pandas.to_datetime(x_t["Month"])
    x_t.set_index("Month", inplace=True)
    x_t.columns = ["passengers"]
    x_t.index.name = "Fecha"
    return  x_t

def format_data(data, l=1):
	X, Y = [], []
	for i in range(len(data)-l-1):
		a = data[i:(i+l), 0]   ###i=0, 0,1,2,3
		X.append(a)
		Y.append(data[i + l, 0])
	return np.array(X), np.array(Y)

def split_data(x_t, split):
    split_data = int(split*len(x_t))
    train, test = x_t[:split_data], x_t[split_data:]
    return train, test

def scale_data(x_t):
    normalizar = MinMaxScaler()
    normalizar.fit(x_t)
    x_t = normalizar.transform(x_t)
    return x_t, normalizar

def add_noise(x_t):
    ruido = np.random.normal(0,0.02, x_t.shape[1])
    x_t[:1,] += ruido 
    return x_t

def LSTM_model(n_input):
    model = Sequential()
    model.add(LSTM(20,return_sequences=True,input_shape=(n_input,1)))
    model.add(LSTM(20,return_sequences=True))  
    model.add(LSTM(20,return_sequences=True)) 
    model.add(LSTM(20))   
    model.add(Dropout(0.15))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    
    return model


if __name__ == '__main__':
    
    x_t = item_1()
    #item 2: Formateo despues
    x_t, normalizar  = scale_data(x_t)
    
    #item 3
    x_t = add_noise(x_t)
    
    #item 4
    train, test = split_data(x_t, 0.7)
    
    #item 5 aca formateo
    l = 4
    x_train, y_train = format_data(train, l)
    x_test, y_test = format_data(test, l)
    
    x_train =   x_train.reshape(x_train.shape[0], l, 1)
    x_test  =   x_test.reshape(x_test.shape[0], l, 1)

    #item 6
    
    model = LSTM_model(l)
    model.fit(x_train,y_train,  validation_data=(x_test,y_test),
              epochs=100,       batch_size=1,   verbose=1)

    train_predict=model.predict(x_train)
    test_predict=model.predict(x_test)

    train_predict=normalizar.inverse_transform(train_predict)
    test_predict=normalizar.inverse_transform(test_predict)
    
    from sklearn.metrics import mean_squared_error
    
    mse_train=np.sqrt(mean_squared_error(y_train,train_predict))    
    mse_test=np.sqrt(mean_squared_error(y_test,test_predict)) 
    
    look_back=l
    trainPredictPlot = np.empty_like(x_t)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(x_t)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(x_t)-1, :] = test_predict
    # plot baseline and predictions
    plt.xlabel("Month")
    plt.ylabel("Passengers")
    plt.title("Passengers Travelled")
    plt.plot(normalizar.inverse_transform(x_t)) #original data
    plt.plot(testPredictPlot) #test prediction
    plt.show()