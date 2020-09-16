import numpy as np
from keras import datasets
np.random.seed(40)
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})

# def accuracy(y_pred, y_true):
#     #y_pred = np.argmax(scores, axis=0)
#     acc = (y_pred==y_true)
#     return np.mean(acc)


def MSE(scores, y_true):
    mse = np.mean(np.sum((scores-y_true)**2, axis=0))
    return mse

def grad_mse(scores, y_true):
    return 2*(scores-y_true)

def sigmoid(x):
    try:
        exp = 1 + np.exp(-1*x)
    except RuntimeWarning:
        print("F")
        exit()
    return 1/exp

def grad_sigmoid(x):
    return np.exp(-x)*(sigmoid(x)**2)



class Classifier(object):
    def __init__(self,  eta         =   0.01, 
                        tolerance   =   9e-1, 
                        epochs      =   10,
                        use_bias    =   False,
                        batch_size  =   2,
                        lambda_L2   =   0.5):

        self.eta        =   eta
        self.tolerance  =   tolerance
        self.epochs     =   epochs
        self.batch_size =   batch_size
        self.use_bias   =   use_bias
        self.lambda_L2  =   lambda_L2

    def predict(self, scores):
        ajax= np.argmax(scores, axis=0)
        output= np.zeros_like(scores)  
        output[ajax, np.arange(scores.shape[1])]=1
        return output

    def accuracy(self, y_pred, y_true):
        acc = (y_pred==y_true)
        return np.mean(acc)
    
    def loss_function(self, scores, y_true):
        pass

    def flattening(self, x, y ):
        X= np.copy(x) 
        X= np.reshape(X, (X.shape[0], np.prod(x.shape[1:])))
        X= X/255 - X.max()/255
        
        X= np.hstack(( (np.ones((len(X),1) )), X)) 
       
        Y = np.zeros(shape=(self.n_clasifi,y.shape[0]))
        y_aux  = np.copy(y).reshape(y.shape[0])

        Y[y_aux, np.arange(Y.shape[1])]=1

        return X,Y.T

    def fit(self, x, y, x_t, y_t):
        self.n_clasifi=10

        self.x,      self.y      = self.flattening(x,y)
        self.x_test, self.y_test = self.flattening(x_t,y_t)

        self.acc_vect = np.zeros(self.epochs)
        self.loss_vect= np.zeros(self.epochs)
        self.pres_vect= np.zeros(self.epochs)

        iter_batch= int(x.shape[0]/self.batch_size)

        w1 = np.random.uniform(-0.0001, 0.0001, size=(100,(self.x.shape[1])))
        w2 = np.random.uniform(-0.001, 0.001, size=((10, (w1.shape[0]+1))))

        for it in range(self.epochs):          
            for it_ba in range(iter_batch):
                
                index   =   np.random.randint(0, x.shape[0], self.batch_size)
                x_batch =   self.x[index]
                y_batch =   self.y[index]
                
                ####Forward####
                S1= sigmoid(np.dot(x_batch, w1.T))                
                S1= np.hstack(((np.ones((len(S1),1) ),S1))) 
                
                S2= np.dot(S1,w2.T)

                ####regularization##
                reg1= 0.0#np.sum(w1*w1)
                reg2= 0.0#np.sum(w2*w2)

                reg = reg1+reg2

                #loss  and acc
                
                self.loss_vect[it] += MSE(S2,y_batch) + 0.5*self.lambda_L2*reg

                S2_out = self.predict(S2)

                self.acc_vect[it]  += self.accuracy(S2_out, y_batch)

                ###"Backguard"
                grad = grad_mse(S2, y_batch)/self.batch_size #+ reg2
                
                #Capa 2
                gradw2  = np.dot(grad.T, S1) + self.lambda_L2*w2
                grad    = np.dot(grad, w2)

                #Capa 1

                grad_sig = grad_sigmoid(S1)                
                grad = grad*grad_sig + reg1
                grad = np.delete(grad, (0), axis=1)
                gradw1 = np.dot(grad.T, x_batch) +  self.lambda_L2*w1


                w1+= -self.eta*(gradw1)
                w2+= -self.eta*(gradw2)

            self.loss_vect[it]/=iter_batch
            self.acc_vect[it]/=iter_batch*0.01

            S1_test= sigmoid(np.dot(self.x_test,w1.T))

            S1_test= np.hstack(((np.ones((len(S1_test),1) ),S1_test)))
            S2_test= np.dot(S1_test, w2.T)

            S2_tout= self.predict(S2_test)

            self.pres_vect[it] = 100*self.accuracy(S2_tout, self.y_test)

            print("Epoch: {} - pres:{:.4} - loss:{:.4} - acc:{:.4}".format(it, self.pres_vect[it],self.loss_vect[it], self.acc_vect[it]))




class layer(object):
    def __init__(self,  input_size=16, 
                        output_size=1,  
                        bias=False, 
                        activation=None, 
                        name='No name'):
        """
        La función de activación ya está necesariamente vectorizada
        y tiene que se ser inicializa o  sino  muere
        """
        self.input_size = input_size
        self.output_size=output_size
        self.bias       = bias
        self.activation =activation
        self.name       =name
        self.weight_init()

        if activation==None:
            print("Dame la función de  activacion\n")
            exit()

    def weight_init(self, initializer= np.random.uniform ):
        self.w = initializer(-1,1,shape=(self.input_size, self.output_size))
        
        if self.bias==True:
            self.w = np.hstack(( (initializer(-1,1, shape=self.output_size), self.w))) 

    def local_gradient(self):
        pass
    
    def call(self, inputs):
        inputs_copy= np.copy(inputs)
        
        if self.bias==True:
            inputs_copy = np.hstack(( (np.ones((len(inputs_copy),1) )), inputs_copy)) 
        
        return self.activation(np.matmul(inputs_copy, weight_init) + self.b)  
        
def ejer3():
    proto= Classifier(epochs =60,
                      batch_size=32,
                      eta    = 0.01,
                      lambda_L2=0.000)

    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    proto.fit(x_train, y_train, x_test, y_test)

    plt.figure(1)
    plt.plot(proto.acc_vect, label="Acc")
    plt.plot(proto.pres_vect, label="Pres")
    plt.legend(loc=0)

    plt.figure(2)
    plt.plot(proto.loss_vect, label="loss")
    plt.legend(loc=0)
    plt.show()

    pass

def main():
    ejer3()
    pass


if __name__ == '__main__':
    main()
    