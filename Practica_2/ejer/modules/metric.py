import numpy as np 

def accuracy(y_pred,y_true):
    acc = (y_pred==y_true)
    return np.mean(acc)

def accuracy_xor(y_pred,y_true):
    y_pred[y_pred>0]=1
    y_pred[y_pred<=0]=-1
    acc = (y_pred==y_true)
    return np.mean(acc)

def mse(y_pred,y_true):
    return np.mean(np.sum((scores-y_true)**2, axis=0))