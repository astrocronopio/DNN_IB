import tensorflow.keras.datasets  as datasets#import cifar10
import numpy as np 

class knearest(object):
    def __init__(self):
        self.x=None
        self.y=None
    
    def train(self, x, y):
        self.im_shape =x.shape[1:]
        self.x = np.reshape(x, (x.shape[0], np.prod(self.im_shape)))
        self.y = y

    def predict(self, x):
        assert self.x is not None, 'Hay que entrenar primero'
        yp = np.zeros(x.shape[0])#, dtype=np.uint8)
        
        for idx in range(x.shape[0]):
            norm = np.linalg.norm(self.x - x[idx].ravel(), axis=1)
            idmin= np.argmin(norm)
            yp[idx] = self.y[idmin]
        
        return  yp


def ejer3():
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    print('x_train shape: ', x_train.shape)
    print(x_train.shape[0], 'ejemplos de entrenamiento')
    print(x_test.shape[0], 'ejemplos para probar')
    
    model = knearest()
    model.train(x_train, y_train)
    yp=model.predict(x_test[:10])

    print(yp)

def main():
    ejer3()
    pass

if __name__ == '__main__':
    main()
    