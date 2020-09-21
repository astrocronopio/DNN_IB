#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 19,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Computer Modern Roman']})

cmap = plt.get_cmap('jet',15)

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
    """ n= dimension, N= Cantidad de elementos de entrenamiento """
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
    ejemplos      = np.arange(100,550,1)
    dimensiones   = np.array([50,100,150,200,250,300])

    i400= np.where(ejemplos==450)[0][0]
    i300= np.where(ejemplos==350)[0][0]
    i500= np.where(ejemplos==500)[0][0]

    a_error=np.zeros((len(dimensiones), len(ejemplos)))
    y_error=np.zeros((len(dimensiones), len(ejemplos)))


    for i in range(len(dimensiones)):
        e = np.zeros((2,len(ejemplos)))
        for j in range(len(ejemplos)):
            if ejemplos[j]>(dimensiones[i]+30):
                y_error[i][j], a_error[i][j] = regresion_lineal_numpy(dimensiones[i], ejemplos[j])
            else: continue

    left, bottom, width, height = [0.45, 0.59, 0.35, 0.32]

    fig1, ax11 = plt.subplots()    
    #ax21 = fig1.add_axes([left, bottom, width, height])
    ax11.set_title("MSE entre $y_{exacto}$ e $y_{esperado}$")
    ax11.set_ylabel("MSE")
    ax11.set_xlabel("Ejemplos")
    
    fig2, ax12 = plt.subplots()
    #ax22 = fig2.add_axes([left, bottom, width, height])
    ax12.set_title("MSE entre $a_{exacto}$ y $a_{esperado}$")
    ax12.set_ylabel("MSE")
    ax12.set_xlabel("Ejemplos")
    
    for i in range(len(dimensiones)):
        ax11.plot(ejemplos[ejemplos>(dimensiones[i]+30)],y_error[i][y_error[i]>0], label="d={}".format(dimensiones[i]), c=cmap(i))
        ax12.plot(ejemplos[ejemplos>(dimensiones[i]+30)],a_error[i][a_error[i]>0], label="d={}".format(dimensiones[i]), c=cmap(i))

    plt.figure(22)

    plt.plot(dimensiones, y_error[:,i300], label="350", c="red", alpha=0.6)
    plt.scatter(dimensiones, y_error[:,i300], c="red", alpha=0.6)

    plt.plot(dimensiones, y_error[:,i400], label="450", c="blue", alpha=0.6)
    plt.scatter(dimensiones, y_error[:,i400], c="blue", alpha=0.6)

    plt.plot(dimensiones, y_error[:,i500], label="500", c="black", alpha=0.6)
    plt.scatter(dimensiones, y_error[:,i500], c="black", alpha=0.6)

    plt.ylabel("MSE")
    plt.xlabel("Dimensión")
    plt.legend(loc=0)
    plt.savefig("ejer_1_mse_y_ejemplos_fijo.pdf")

    plt.figure(24)

    plt.plot(dimensiones, a_error[:,i300], label="350", c="blue", alpha=0.6)
    plt.scatter(dimensiones, a_error[:,i300], c="blue", alpha=0.6)
    
    plt.plot(dimensiones, a_error[:,i400], label="450", c="red", alpha=0.6)
    plt.scatter(dimensiones, a_error[:,i400], c="red", alpha=0.6)

    plt.plot(dimensiones, a_error[:,i500], label="500", c="black", alpha=0.6)
    plt.scatter(dimensiones, a_error[:,i500], c="black", alpha=0.6)
    
    plt.ylabel("MSE")
    plt.xlabel("Dimensión")
    plt.savefig("ejer_1_mse_a_ejemplos_fijo.pdf")

    ax11.legend(loc=0)


    ax12.legend(loc=0)
    plt.legend(loc=0)

    fig1.savefig("ejer_1_mse_y_ejemplos.pdf")
    fig2.savefig("ejer_1_mse_a_ejemplos.pdf")

    plt.show()

def main( ):
    ejer1()
    pass

if __name__ == '__main__':
    main()