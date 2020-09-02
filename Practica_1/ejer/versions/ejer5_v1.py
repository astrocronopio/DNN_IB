import numpy as np 
import tensorflow.keras.datasets  as datasets

import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})

def activacion(W,x, y=[]):
    return np.matmul(W,x)

def VSMclassifier(W,x,y):
    yp=activacion(W,x)
    return np.heaviside(yp - yp[y])
    pass

def VSMclassifier_gradient(W,x,y,L):
    print("HOla")
    return None
    """ print(np.dot(L, x))
    return np.dot(L, x)
 """
#####################################3
""" def softmaxclassifier(W,x,y):
    yp = activacion(W,x)
    yp -= np.max(yp)
    expo= np.exp(yp)
    return -yp + np.log(np.sum(expo)) 

def softmaxclassifier_gradient(W,x,y,L):
    return [] """

class LinearClassifier(object):
    def __init__(self, eta=0.01, tolerance=9e-1, epochs=2):
        self.x=None
        self.y=None
        self.eta=eta
        self.tolerance=tolerance
        self.epochs=epochs
        self.loss_func=None
        #self.loss_func_gradient=None

    def predict(self, x):
        return np.argmax(np.dot(self.W, x))
    
    def fit(self, x, y, use_bias=False):
        n_clasifi=10
        self.im_shape =x.shape[1:]
        self.x = np.reshape(x, (x.shape[0], np.prod(self.im_shape)))
        
        if (use_bias==True):
            np.append(self.x, 1)
        
        self.W = np.random.uniform(-1,1,size=(n_clasifi, self.x.shape[1])) 
        self.y = y

        #print(y.shape, self.x.shape, self.W.shape)

        error=np.zeros(self.epochs)
        
        for iteracion in range(self.epochs):
            for index in range(len(y)):
                error[iteracion] += self.bgd(self.x[index], self.y[index])

        return error

    def sample_data(self, x_batch, y_batch, batch_size):
        return self.x[0], self.y[0]


    def bgd(self,x_batch, y_batch):
        error=[]
        delta= np.inf
        #x_batch, y_batch = self.sample_data(batch_size)
        while np.any(delta>self.tolerance):
            old=self.loss_func(self.W, x_batch , y_batch)
            w_grads = self.loss_func_gradient(self.W, x_batch, y_batch, old)
            #print(w_grads)
            #self.W+= -self.eta*w_grads
            #print("hi")
            delta = np.abs(old - self.loss_func(self.W,x_batch, y_batch))
            error.append(delta)
        return 1#np.sqrt(np.sum(( activacion(self.W,x_batch,0) - y_batch)))

""" class SMC(LinearClassifier):
    def __init__(self):
        super(SMC, self).__init__()
        self.loss_func=softmaxclassifier
        self.loss_func_gradient=softmaxclassifier_gradient
 """

class VSM(LinearClassifier):
    def __init__(self, *args):
        super(VSM, self).__init__()
        self.loss_func=VSMclassifier
        self.loss_func_gradient=VSMclassifier_gradient
        
        

def ejer5():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.astype(np.float)
    print('Dimensiones del set de entrenamiento ', x_train.shape)
    print(x_train.shape[0], 'ejemplos de entrenamiento')
    print(x_test.shape[0], 'ejemplos para probar')
    
    model_smc = SMC()
    errormodel_smc= model_smc.fit(x_train, y_train, use_bias=True)    
    plt.plot(errormodel_smc)

    #model_vsm = VSM()
    #errormodel_vsm= model_vsm.fit(x_train, y_train, use_bias=True)     
    #plt.plot(errormodel_vsm)

    plt.show()
    pass

def main():
    ejer5()
    pass

if __name__ == '__main__':
    main()
    