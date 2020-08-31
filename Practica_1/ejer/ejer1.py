import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 24,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Times']})


def solucion_exacta(conjunto_x, conjunto_y):
    pinversa   =  np.linalg.pinv(conjunto_x) #np.transpose(conjunto_x).dot(conjunto_x) #.dot(np.transpose(conjunto_x))
    soluciones =  np.dot(pinversa, conjunto_y)
    #(XtX)^{-1} Xt #np.linalg.solve(conjunto_x, conjunto_y) 
    return soluciones

def min_squared(y,y_i):
    try:
        return np.sqrt(np.sum((y-y_i)**2)/len(y))
    except TypeError:
        return (y-y_i)**2


def ejemplos(n,N):
    a = np.random.uniform(-4,4,size=(n+1))
    conjunto_x , y= [], []

    for _ in range(N):
        x = np.random.uniform(-2,2,size=(n+1))
        x[-1]=1 #asociado al término independiente
        conjunto_x.append(x)
        y.append(np.dot(a,x) + np.random.uniform(-1,1))
        pass
    return conjunto_x, y, a

def regresion_lineal_numpy(n, N):
    #N=int(0.9*n)
    print(N)

    conjunto_x, conjunto_y, a = ejemplos(n,N)
    print(np.shape(conjunto_x), np.shape(conjunto_y))
    a_2 = solucion_exacta(conjunto_x, conjunto_y)

    x = np.random.uniform(-2,2,size=(n+1))
    x[-1]=1

    solucion_y = np.dot(x, a_2)
    solucion_exacta_y = np.dot(x, a) 

    error = min_squared(solucion_y, solucion_exacta_y)
    error2 = min_squared(a_2, a)
    return error, error2


def ejer1():
    dimensiones = np.arange(5,150,1)

    porcentaje_vector = [ 0.25, 0.5, 1, 1.1, 2, 2.5]

    for porcentaje in porcentaje_vector:
        e1, e2= [], []
        aux1, aux2= 0,0
        for dimension in dimensiones:
            aux1, aux2 = regresion_lineal_numpy(dimension, int(porcentaje*dimension))
            e1.append(aux1)
            e2.append(aux2)
        #print(e)
        plt.figure(1)
        plt.plot(dimensiones, e1)
        plt.figure(2)
        plt.plot(dimensiones, e2)
    plt.show()

    pass

def main( ):
    ejer1()
    pass

if __name__ == '__main__':
    main()