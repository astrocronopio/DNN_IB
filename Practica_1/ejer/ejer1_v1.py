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
    soluciones = np.linalg.solve(conjunto_x, conjunto_y) 
    return soluciones

def perceptrones(conjunto_x, conjunto_y, iteraciones):
    pass

def min_squared(y,y_i):
    return np.sqrt(np.sum((y-y_i)**2)/len(y))


def ejemplos(n,N):
    a = np.random.uniform(-4,4,size=n)
    b = np.random.uniform(-4,4)
    conjunto_x , y= [], []
    for i in range(N):
        x = np.random.uniform(-4,4,size=n)
        np.append(x,1) #asociado al término independiente
        conjunto_x.append(x)
        y.append(np.dot(a,x) + b + np.random.uniform(-10,10))
        pass
    return conjunto_x, y, np.append(a,b)

def regresion_lineal_numpy(n):
    N=10
    conjunto_x, conjunto_y, a = ejemplos(n,N)
    a_2 = solucion_exacta(conjunto_x, conjunto_y)

    solucion_y = np.dot(conjunto_x, a_2)
    error = min_squared(solucion_y, conjunto_y)
    return error


def regresion_lineal_perceptron(n):
    N = int(n*0.5)
    iteraciones = 100

    conjunto_x, conjunto_y, a = ejemplos(n, N)
    #a_1 = perceptrones(conjunto_x, conjunto_y, iteraciones)
    a_2 = solucion_exacta(conjunto_x, conjunto_y)

    solucion_y = np.dot(conjunto_x, a_2)
    '''     a_1= a_2
    try: 
        error_1, error_2 = min_squared(a_1, a), min_squared(a_2, b)
    except TypeError:
        return np.nan, np.nan '''

    error_1 = min_squared(solucion_y, conjunto_y)
    return error_1  # , error_2

def ejer1():
    dimensiones = np.arange(5,80,1)
    e= []
    for dimension in dimensiones:
        e.append(regresion_lineal_numpy(dimension))
    print(e)
    plt.plot(dimensiones, e)
    plt.show()

    pass

def main( ):
    ejer1()
    pass

if __name__ == '__main__':
    main()
    
