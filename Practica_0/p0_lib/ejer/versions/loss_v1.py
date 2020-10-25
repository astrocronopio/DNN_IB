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
        super()
        mse = np.mean(np.sum((scores-y_true)**2, axis=0))
        return mse

    def gradient(self, scores, y_true):
        """La primera componente es siempre el tamaño de batch"""
        return 2*(scores-y_true)/y_true.shape[0]       

class cross_entropy(loss):
    def __call__(self, scores, y_true):
        super()
        scores-= np.max(scores,axis=0)

        ind= np.arange(y_true.shape[0], dtype=np.int)
        y = np.argmax(y_true, axis=0)

        expo    =   np.exp(scores)
        expo_sum=   np.sum(expo, axis=0)
        
        self.diff    =   expo/expo_sum
        self.diff[y,ind] += -1

        loss = np.mean(-scores[y,ind] + np.log(np.sum(expo, axis=0)))

        return loss

    def gradient(self, scores, y_true):  
        super()  
        """La primera componente es siempre el tamaño de batch"""
        return self.diff/y_true.shape[0]
