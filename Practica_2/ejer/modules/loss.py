import numpy as np

class loss(object):
    def __init__(self):
        pass
    def __call__(self, scores, y_true):
        pass
    def gradient(self, scores, y_true):
        pass    


class MSE(loss):
    def __call__(self, scores, y_true):
        mse = np.mean(np.sum((scores-y_true)**2, axis=0))
        return mse

    def gradient(self, scores, y_true):
        """La primera componente es siempre el tamaño de batch"""
        return 2*(scores-y_true)/y_true.shape[0]       

