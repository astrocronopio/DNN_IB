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
        mse = np.mean(np.sum((scores-y_true)**2, axis=1))
        return mse

    def gradient(self, scores, y_true):
        """La primera componente es siempre el tamaño de batch"""
        return 2*(scores-y_true)/y_true.shape[0]       


class cross_entropy(loss):
    def __call__(self, scores, y_true):
        ind= np.arange(scores.shape[0], dtype=np.int)
        scores -= np.max(scores,axis=1)[:, np.newaxis]

        expo    =   np.exp(scores)
        y = np.argmax(y_true, axis=1)

        loss = np.mean(-scores[ind,y] + np.log(np.sum(expo, axis=1)))
        return loss

    def gradient(self, scores, y_true): 
        ind= np.arange(scores.shape[0], dtype=np.int)
        #scores -= np.max(scores,axis=1)[:, np.newaxis] 
        #Lo hago en la __call__

        expo    =   np.exp(scores)
        y = np.argmax(y_true, axis=1)

        diff    =   expo/np.sum(expo, axis=1)[:,np.newaxis]
        
        diff[ind,y] += -1  
        """La primera componente es siempre el tamaño de batch"""
        return diff/y_true.shape[0]
