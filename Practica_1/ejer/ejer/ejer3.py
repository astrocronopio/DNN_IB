import tensorflow.keras.datasets  as datasets
import numpy as np 

class knearest(object):
    def __init__(self):
        self.x=None
        self.y=None
    
    def entrenar(self, x, y):
        self.im_shape =x.shape[1:]
        self.x = np.reshape(x, (x.shape[0], np.prod(self.im_shape)))
        self.y = y

    def predecir(self, x, k=1):
        assert self.x is not None, 'Hay que entrenar primero'
        yp = np.zeros(x.shape[0])

        for idx in range(x.shape[0]):
            norm = np.linalg.norm(self.x - x[idx].ravel(), axis=1)
            idmin = np.argpartition(norm,k)[:k] #los indices de los k menores valores
            kvecinos = self.y[idmin].ravel() 
            clasificacion = np.bincount(kvecinos) #Cuenta cuantas veces se repite cada vecino
            yp[idx] = np.argmax(clasificacion) # El vecino que mas se repite
        
        return  yp

def knn_implementacion(data, prueba, k=1):
    (x_train, y_train), (x_test, y_test) = data#mnist.load_data()
    x_train = x_train.astype(np.float)
    print('Dimensiones del set de entrenamiento ', x_train.shape)
    print(x_train.shape[0], 'ejemplos de entrenamiento')
    print(x_test.shape[0], 'ejemplos para probar')
    
    model = knearest()
    model.entrenar(x_train, y_train)
    yp=model.predecir(x_test[:prueba], k)
    counter=0

    for a,b in zip(yp,y_test):
        if (a==b): counter+=1

    print("\n{}% probando con {} ejemplos\n".format(counter*100./len(yp), prueba))
    return model

def ejer3():
    print("Con el MNIST: ")
    knn_implementacion(datasets.mnist.load_data(), 20)
    print("Con el CIFAR-10: ")
    knn_implementacion(datasets.cifar10.load_data(), 20)
    pass

def main():
    ejer3()

if __name__ == '__main__':
    main()
    