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
    x_t.index.name = "date"
    return  x_t

def item_2(x_t, l):
    normalizar = MinMaxScaler()
    normalizar.fit(x_t)
    x_t = normalizar.transform(x_t)
    y_t = x_t[:-l]
    x_t = x_t[:len(y_t)]
    return x_t, y_t, normalizar

def item_3(x_t):
    ruido = np.random.normal(0,0.02, x_t.shape[1])
    x_t[:1,] += ruido 
    return x_t

def item_4(x_t, y_t, split):
    split_data = int(split*len(x_t))
    
    train_x_t, test_x_t = x_t[:split_data], x_t[split_data:]
    train_y_t, test_y_t = y_t[:split_data], y_t[split_data:]
    return train_x_t,  train_y_t, test_x_t, test_y_t

def item_5(x_train, y_train, n_input):
    print(x_train.shape)
    print(y_train.shape)
    print(n_input)
    
    time_sequence_train = TimeseriesGenerator(x_train, y_train, length=n_input)
    return time_sequence_train

def item_6(time_series, n_input):
    model = Sequential()
    
    model.add(LSTM(10, activation='relu', input_shape=(n_input, 1)))
    model.add(LSTM(10, activation='relu', input_shape=(n_input, 1)))
    model.add(LSTM(10, activation='relu', input_shape=(n_input, 1)))
    model.add(LSTM(10, activation='relu', input_shape=(n_input, 1)))
    
    model.add(Dropout(0.15))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    model.fit(time_series, epochs=90, verbose= 1)
    model.summary()
    
    return model


if __name__ == '__main__':
    x_t = item_1()
    x_t, y_t, normalizar = item_2(x_t, 12)
    
    x_t = item_3(x_t)
    
    x_train, y_train, x_test, y_test = item_4(x_t, y_t, 0.6)
    
    n_input = len(x_train)
    
    time_series = item_5(x_train, y_train, n_input)
    
    model = item_6(time_series, n_input)
    exit()
            
    pred_list = []
    batch = x_train[-n_input:].reshape((1, n_input, 1))
    
    for i in range(n_input):   
        pred_list.append(model.predict(batch)[0]) 
        batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)
        
    df_predict = pandas.DataFrame(  normalizar.inverse_transform(pred_list),
                                    index=x_t[-n_input:].index, 
                                    columns=['Prediction']
                                    )

    df_test = pandas.concat([x_t, df_predict], axis=1)
    
    plt.figure(figsize=(20, 5))

    plt.plot(df_test.index, df_test['passengers'])
    plt.plot(df_test.index, df_test['Prediction'], color='r')
    plt.legend(loc='best', fontsize='xx-large')
    plt.xticks(fontsize=18, color= "white")
    plt.yticks(fontsize=16, color= "white")

    plt.show()