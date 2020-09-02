import numpy as np 
import tensorflow.keras.datasets  as datasets

class LinearClassifier(object):
    def __init__(self, *args):
        super(LinearClassifier, self).__init__(*args))

    def loss_gradient(self):

    
    def predict(self):
        pass
    
    def fit(self):
        pass


def bgd(tolerance=1e-3):
    delta= np.inf
    while delta>tol:
        data_batch = sample_data(training_data, batch_size)
        old=loss_fun(data_batch,weights)
        w_grads = evaluate_gradient(loss_fun, data_batch, weights)
        weights+= -step_size*w_grads
        delta = np.abs(old - loss_fun(data_batch, weights))