import numpy as np 

def accuracy(y_pred,y_true):
    acc = (y_pred==y_true)
    pass

def mse(y_pred,y_true):
    return np.mean(np.sum((scores-y_true)**2, axis=0))