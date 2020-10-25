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

def accuracy(scores, y_true):
    y_pred = np.argmax(scores, axis=0)
    acc = (y_pred==y_true)
    return np.mean(acc)


def MSE(scores, y_true):
    mse = np.mean(np.sum((scores-y_true)**2, axis=1))
    return mse

def grad_mse(scores, y_true):
    return 2*(scores-y_true)

def sigmoid(x):
    exp = 1 + np.exp(-1*x)
    return 1/exp

def grad_sigmoid(x):
    return np.exp(-x)*(sigmoid(x))**2


#Preprosesado

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x = np.copy(x_train[:]) 
x = np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
#x/=255 #Porque las imagenes del cifar10 y mnist varian hasta 255

y=  np.copy(y_train[:])
y.reshape(x.shape[0])

y_2 = np.zeros(shape=(10,y.shape[0]))
y_2[y,np.arange(y_2.shape[1])]=1
y=y_2
print(y.shape)


x_t=np.copy(x_test) 
x_t=np.reshape(x_t, (x_t.shape[0], np.prod(x_t.shape[1:])))
#x_t=x_t/255

y_t=  np.copy(y_test)
y_t.reshape(y_t.shape[0])

#Restamos la media del batch
x  = x/255 - 1
#print("xmax: ", x.max())
x= np.hstack(( (np.ones((len(x),1) )), x)) 

x_t= x_t/255 -1 # np.max(x_t)
x_t= np.hstack(( (np.ones((len(x_t),1) )), x_t)) 

w1 = np.random.uniform(-0.1, 0.1, size=(100,(x.shape[1]) ))
print("w1: ", w1.shape)

w2 = np.random.uniform(-0.1, 0.1, size=((10, (w1.shape[0]+1))))
print("w2: ", w2.shape)

epochs=15
batch_size=32
lr= 0.001

lambda_L2=0.1

acc_vect= np.zeros(epochs)
loss_vect=np.zeros(epochs)
pres_vect=np.zeros(epochs)

iter_batch= int(x.shape[0]/batch_size)

for it in range(epochs):
    print("Epoca: ",it)
    loss, acc=0,0
    for it_ba in range(iter_batch): #WHYYYYYYYYYYYYYYY uwuwuuwuwuw
        index   =   np.random.randint(0, x.shape[0], batch_size)
        x_batch =   x[index]#self.x[index:final]
        y_batch =   y[:,index]

        #print("Forward")
        
        act= np.dot(w1, x_batch.T)
        #print("act: ", act.max())
        try:
            S1= sigmoid(act)
        except RuntimeWarning:
            print("F")
            exit()
        
        
        S1= np.hstack(((np.ones((len(S1.T),1) ),S1.T))) 

        #print(S1.shape)
        #print("S1 max: ",S1.max())
        
        if S1.max==np.nan: 
            exit()

        S2= np.dot(w2, S1.T)

        #print("regularixzation")
        reg1= np.sum(w1*w1)
        reg2= np.sum(w2*w2)

        reg = reg1+reg2

        #is this loss?
        #print("S2, y_batch", S2, y_batch)

        loss = loss + MSE(S2,y_batch) + 0.5*lambda_L2*reg
        acc  = acc  + accuracy(S2, y_batch)

        #print("Backguard")

        grad = grad_mse(S2, y_batch) #+ reg2

        #Capa 2

        gradw2  = np.dot( grad, S1) + lambda_L2*w2
        grad    = np.dot(w2.T, grad)
        #print(grad.shape)
        #grad    = grad[grad.shape[0]-1:]
        #print(grad.shape)
        #print("gw2: ", gradw2.shape)
        
        #Capa 1

        #print("S1: ", S1.shape)
        #grad_sig = grad_sigmoid(S1.T)

        try:
            grad_sig = grad_sigmoid(S1.T)
        except RuntimeWarning:
            print("F")
            exit()
        
        grad = grad*grad_sig + reg1
        grad = grad[grad.shape[0]-1:]

        gradw1 = np.dot(grad, x_batch) +  lambda_L2*w1
        #print("gw1: ", gradw1.shape)
        
        #update
        w1+= -lr*(gradw1)
        w2+= -lr*(gradw2)
    
    loss_vect[it]=loss/batch_size
    acc_vect[it]=acc/batch_size

    try:
        S1_test= sigmoid(np.dot(w1, x_t.T))
    except RuntimeWarning:
        print("F")
        exit()    
    
    #S1_test= sigmoid(np.dot(w1, x_t.T))
    S1_test= np.hstack(( S1_test.T, (np.ones((len(S1_test.T),1) )))) 
    S2_test= np.dot(w2, S1_test.T)

    #print("S2, y_test", S2.shape, y_t.shape)

    pres_vect[it] = accuracy(S2_test, y_t)

    print("pres: ", pres_vect[it])


plt.figure(1)
plt.plot(acc_vect, label="Acc")
plt.plot(pres_vect, label="Pres")
plt.legend(loc=0)

plt.figure(2)
plt.plot(loss_vect, label="loss")
plt.legend(loc=0)
plt.show()