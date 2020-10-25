import numpy as np 

def accuracy(y_pred,y_true):
    ajax1= np.argmax(y_pred, axis=1)
    ajax2= np.argmax(y_true, axis=1)
    acc = (ajax1==ajax2)
    return np.mean(acc)

def accuracy_xor(y_pred,y_true, y_lim=0.7):
    y_pred[y_pred>y_lim]=1
    y_pred[y_pred<-y_lim]=-1
    acc = (y_pred==y_true)
    return np.mean(acc)

def mse(y_pred,y_true):
    return np.mean(np.sum((scores-y_true)**2, axis=0))