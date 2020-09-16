import numpy as np
import model.layer as layer 
import model.layer as layer 

class Model(object):
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
        ajax= np.argmax(scores, axis=1)
        return ajax#output

    def accuracy(self, y_pred, y_true):
        acc = (y_pred.T==y_true.T)
        return np.mean(acc)
    
    def loss_function(self, scores, y_true):
        pass

    def fit(self, x, y, x_test, y_test):

        self.acc_vect = np.zeros(self.epochs)
        self.loss_vect= np.zeros(self.epochs)
        self.pres_vect= np.zeros(self.epochs)

        iter_batch= int(x.shape[0]/self.batch_size)

        layer1= layer.layer(input_size=x.shape[0], output_size=100,
                            activation= , name='L1')
                        #w1 = np.random.uniform(-0.00001, 0.00001, size=(100,(x.shape[1])))
        
        layer2= layer.layer(input_size=101, output_size=10,
                        activation=, name='No name')
                        #w2 = np.random.uniform(-0.001, 0.001, size=((10, (w1.shape[0]+1))))


        for it in range(self.epochs): 
            loss,acc=0,0         
            for it_ba in range(iter_batch):
                
                index   =   np.random.randint(0, x.shape[0], self.batch_size)
                x_batch =   x[index]
                y_batch =   y[index]
                
                ####Forward####
                S1= sigmoid(np.dot(x_batch, w1.T))                
                S1= np.hstack(((np.ones((len(S1),1) ),S1))) 

                S2= np.dot(S1,w2.T)
                """Hasta acá bien"""
                ####regularization##

                reg1= np.sum(w1*w1)
                reg2= np.sum(w2*w2)
                reg = reg1+reg2

                #loss and acc
                
                loss += MSE(S2,y_batch) + 0.5*self.lambda_L2*reg

                S2_out = self.predict(S2)
                y_batch_out = self.predict(y_batch)

                acc  += self.accuracy(S2_out, y_batch_out)
                
                ###"Backguard"

                grad2 = grad_mse(S2, y_batch)/self.batch_size #+ reg2
                

                #Capa 2
                gradw2  = np.dot(grad2.T, S1) + self.lambda_L2*w2
                grad1    = np.dot(grad2, w2)

                #Capa 1
                grad_sig = grad_sigmoid(S1)                
                grad1 = grad1*grad_sig 
                grad1 = np.delete(grad1, (0), axis=1)
                gradw1 = np.dot(grad1.T, x_batch) +  self.lambda_L2*w1


                w1+= -self.eta*(gradw1)
                w2+= -self.eta*(gradw2)

            self.loss_vect[it]=loss/iter_batch
            self.acc_vect[it]=100*acc/iter_batch

            S1_test= sigmoid(np.dot(x_test,w1.T))

            S1_test= np.hstack(((np.ones((len(S1_test),1) ),S1_test)))
            S2_test= np.dot(S1_test, w2.T)

            S2_tout= self.predict(S2_test)
            y_test_out=self.predict(y_test)

            self.pres_vect[it] = 100*self.accuracy(S2_tout, y_test_out)

            print("Epoch: {}/{} - pres:{:.4} - loss:{:.4} - acc:{:.4}".format(it, self.epochs, self.pres_vect[it],self.loss_vect[it], self.acc_vect[it]))

        