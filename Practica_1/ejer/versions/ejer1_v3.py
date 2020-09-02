import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Times']})


def solucion_exacta(conjunto_x, conjunto_y):
    aux        =  np.matmul(np.transpose(conjunto_x),conjunto_x)
    pinversa   =  np.matmul(np.linalg.inv(aux), np.transpose(conjunto_x))
    soluciones =  np.matmul(pinversa, conjunto_y) 
    return soluciones

def min_squared(y,y_i):
    try:
        return np.sqrt(np.sum((y-y_i)**2)/len(y))
    except TypeError:
        return (y-y_i)**2


def ejemplos(n,N):
    a = np.random.uniform(-4,4,size=(n+1))
    conjunto_x , y  = [], []

    for _ in range(N):
        x    = np.random.uniform(-2,2,size=(n+1))
        x[-1]=1 #asociado al término independiente
        conjunto_x.append(x)
        y.append(np.dot(a,x) + np.random.uniform(-1,1))
        pass
    return conjunto_x, y, a

def regresion_lineal_numpy(n, N):
    promediar=10
    error=0
    error2=0

    for _ in range(10):
        conjunto_x, conjunto_y, a = ejemplos(n,N)

        try:
            a_2 = solucion_exacta(conjunto_x, conjunto_y)
        except np.linalg.LinAlgError:
            return np.nan,np.nan

        x    = np.random.uniform(-2,2,size=(15,n+1))
        x[-1,:]=np.ones(x.shape[1])

        solucion_y        = np.matmul(x, a_2)
        solucion_exacta_y = np.matmul(x, a) 

        error  += min_squared(solucion_y, solucion_exacta_y)
        error2 += min_squared(a_2, a)

    return error/promediar, error2/promediar


def ejer1():
    dimensiones = np.arange(6,150,1)

    porcentaje_vector = [1, 1.1, 1.5, 2, 2.5, 3]
    plt.yscale('log')

    for porcentaje in porcentaje_vector:
        e1, e2= [], []
        aux1, aux2= 0,0
        for dimension in dimensiones:
            aux1, aux2 = regresion_lineal_numpy(dimension, int(porcentaje*dimension))
            e1.append(aux1)
            e2.append(aux2)
        
        plt.figure(1)
        plt.title("MSE entre $y_{exacto}$ y $y_{esperado}$")
        plt.plot(dimensiones, e1, label="{}".format(porcentaje))
        plt.legend(loc=0, ncol=2)
        plt.yscale('log')

        plt.figure(2)
        plt.title("MSE entre $a_{i,exacto}$ y $a_{i}$")
        plt.plot(dimensiones, e2, label="{}".format(porcentaje))
        plt.legend(loc=0, ncol=2)
        plt.yscale('log')

    plt.show()
    """ dimensiones = np.arange(5,150,1)

    for dimen in dimensiones:
        aux1, aux2= 0,0
        e3, e33=[], []
        aux1, aux2 = regresion_lineal_numpy(dimen, 100)
        if(aux1!=np.nan): e3.append(aux1)
        if(aux2!=np.nan): e33.append(aux2)

    print(e33)
    exit()

    plt.figure(3)
    plt.title("MSE entre $y_{exacto}$ y $y_{esperado}$")
    plt.plot(dimensiones, e3, label="100 ejemplos")
    plt.legend(loc=0, ncol=2)
    
    plt.figure(4)
    plt.title("MSE entre $a_{i,exacto}$ y $a_{i}$")
    plt.plot(dimensiones, e33, label="100 ejemplos")
    plt.legend(loc=0, ncol=2)
  
    plt.show()

 """
def main( ):
    ejer1()
    pass

if __name__ == '__main__':
    main()