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

class LinearClassifier(object):
    def __init__(self, eta=0.01, tolerance=9e-1, epochs=10):
        self.eta=eta
        self.tolerance=tolerance
        self.epochs=epochs

    def predict(self, x):
        return np.argmax(self.activacion(x))
    
    def fit(self, x, y, use_bias=False, batch_size=2):
        n_clasifi=10

        self.im_shape =x.shape[1:]
        self.x = np.reshape(x, (x.shape[0], np.prod(self.im_shape))) #a la imagen la hago un vector
        
        if (use_bias==True): np.append(self.x, 1) #Si hay bias, agrego un 1 al final del vector
        
        self.W = np.random.uniform(-0.01,0.011,size=(n_clasifi, self.x.shape[1])) #Inicializa pesos
        self.y = y

        error_loss=np.zeros(self.epochs)
        error_acc =np.zeros(self.epochs)
        
        for iteracion in range(self.epochs):
            index   =   np.random.randint(0,x.shape[0])

            final   =   index+batch_size
            x_batch =   self.x[index:final]
            y_batch =   self.y[index:final]

            L_loss  =   self.bgd(x_batch, y_batch)
            yp      =   self.predict(x_batch)

            error_acc[iteracion] += (yp==y_batch).mean()
            error_loss[iteracion] = L_loss
            print(L_loss)

        return error_loss*batch_size/x.shape[0], error_acc*batch_size/x.shape[0]

    def bgd(self,x_batch, y_batch):
        loss , dW= self.loss_gradient(self.W, x_batch , y_batch)
        self.W+= -self.eta*dW
        return loss

class SVM(LinearClassifier): #Support Vector Machine
    def activacion(self,x):
        return np.dot(x, self.W.T)+ self.delta

    def loss_gradient(self, W,x,y):
        self.delta=-200
        self.lambda_L2 = 0.005
        yp=self.activacion(x)

        diff = yp - y[:,np.newaxis] + self.delta
        diff = diff*np.heaviside(diff,0)
        L2= np.sum(self.W*self.W)
        
        # y tiene las posiciones de la solucion
        diff[np.arange(x.shape[0]), y]=0 
        # es genial porque las puedo usar para forzar el 0 donde debe ir
        
        L=diff.sum(axis=-1) #sumo intra-vector, ahora tengo un [batchsize,(1)]  
        loss=np.mean(L) + 0.5*self.lambda_L2*L2

        diff_solucion = np.zeros_like(self.W)
        diff_ = np.zeros_like(self.W)

        for i in range(x.shape[0]):
            diff_ += np.dot(diff[i][:,np.newaxis], np.transpose(x[i])[np.newaxis,:])
            aux=np.sum(diff[i])
            diff_solucion[y[i]:]+= -aux*x[i]
        
        binary = diff_solucion + diff_
        binary /=x.shape[0]
        binary += self.lambda_L2*self.W

        dW = binary

        return loss, dW

""" class SMC(LinearClassifier): #SoftMax Classifier
    def activacion(self,x):
        return np.dot(x,self.W.T)

    def loss_gradient(self, W,x,y):
        yp=self.activacion(x)
        loss=np.mean(L)
        return loss, dW """

def ejer5():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.astype(np.float)
    print('Dimensiones del set de entrenamiento ', x_train.shape)
    print(x_train.shape[0], 'ejemplos de entrenamiento')
    print(x_test.shape[0], 'ejemplos para probar')
    
    model_SVM = SVM(eta=0.01, epochs = 100)
    error_loss_1, error_acc_1= model_SVM.fit(x_train[:100], y_train[:100], batch_size=4, use_bias=True)   

    plt.figure(1)
    plt.title('loss')
    plt.plot(error_loss_1)
    
    plt.figure(2)
    plt.title('acc')
    plt.plot(error_acc_1)
    #plt.plot(error_loss_2)  
    plt.show()

""" 
    model_SMC = SMC(eta=0.1, epochs = 100)
    error_loss_2, error_acc_2= model_SMC.fit(x_train[:5], y_train[:5])  

    plt.figure(1)
    plt.title('loss')
    plt.plot(error_loss_1)
    plt.plot(error_loss_2)

    plt.figure(2)
    plt.title('acc')
    plt.plot(error_acc_1)
    plt.plot(error_acc_2)

    plt.show() """
    
def main():
    ejer5()
    pass

if __name__ == '__main__':
    main()
    