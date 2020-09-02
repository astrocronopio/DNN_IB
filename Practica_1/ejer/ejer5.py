import numpy as np 
import tensorflow.keras.datasets  as datasets

def activacion(W,x,b, y=[]):
    return np.matmul(W,x)+b

def VSMclassifier(W,x,b,y):
    yp=activacion(W,x,b)
    return np.heaviside(yp - y)
    pass

def softmaxclassifier(W,x,b,y):
    yp=activacion(x,W,0)
    yp -= np.max(f)
    return -yp + np.log(np.sum(y)) 


class LinearClassifier(object):
    def __init__(self):
        self.x=None
        self.y=None
        self.loss_func=None#activacion

    def loss_gradient(self, data):
        pass
    
    def predict(self, x):
        pass
    
    def fit(self, x, y, eta=0.01, tolerance=1e-3, epochs=100, use_bias=False):
        self.im_shape =x.shape[1:]
        self.x = np.reshape(x, (x.shape[0], np.prod(self.im_shape)))
        
        if (use_bias==True):
            np.append(self.x, 1)
        
        self.W = np.random.uniform(-1,1,size=(x.shape[1],y.shape[1])) 
        self.y = y

        self.eta=eta
        self.tolerance=tolerance
        
        error=np.zeros(epochs)
        for e in error:
            e=bdg()
        return error

    def sample_data(self, batch_size):
        


    def bgd(batch_size=8):
        error=[]
        delta= np.inf
        while delta>tol:
            x_batch, y_batch = sample_data(batch_size)
            old=self.loss_fun(self.W,x_batch,self.b, y_batch )
            w_grads = loss_gradient(x_batch, y_batch)
            self.W+= -self.eta*w_grads
            delta = np.abs(old - loss_fun(self.W,x_batch, self.b, y_batch))
            error.append(delta)

class SMC(LinearClassifier):
    def __init__(self):
        super(SMC, self).__init__()
        self.loss_func=softmaxclassifier


class VSM(LinearClassifier):
    def __init__(self, *args):
        super(VSM, self).__init__()
        self.loss_func=VSMclassifier
        
        

def ejer5():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.astype(np.float)
    print('Dimensiones del set de entrenamiento ', x_train.shape)
    print(x_train.shape[0], 'ejemplos de entrenamiento')
    print(x_test.shape[0], 'ejemplos para probar')
    
    model_smc = SMC()
    model_vsm = VSM() 
    pass

def main():
    ejer5()
    pass

if __name__ == '__main__':
    main()
    