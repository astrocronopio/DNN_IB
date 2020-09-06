#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Computer Modern Roman']})

cmap = plt.get_cmap('rainbow',16)

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


def plot_me(n, dimensiones, e1, title, label, save_file, i):
    plt.figure(n)
    plt.title(title)
    plt.plot(dimensiones[0:len(e1)], e1, label=label, c=cmap(i))
    plt.legend(loc=0, ncol=2)
    plt.xlabel("Ejemplos")
    plt.ylabel("MSE")
    plt.yscale('log')
    plt.savefig(save_file)

# def ejer1():
#     dimensiones = np.arange(6,150,1)
#     porcentaje_vector = [1, 1.1, 1.5, 2, 2.5, 3]
#     i=-1
#     for porcentaje in porcentaje_vector:
#         i+=1
#         e1, e2= [], []
#         aux1, aux2= 0,0
#         for dimension in dimensiones:
#             aux1, aux2 = regresion_lineal_numpy(dimension, int(porcentaje*dimension))
#             e1.append(aux1)
#             e2.append(aux2)
        
#         plot_me(1, dimensiones, e1, "MSE entre $y_{exacto}$ e $y_{esperado}$", 
#                 "{}".format(porcentaje), "ejer_1_mse_y_porcentaje.pdf",i)

#         plot_me(2, dimensiones, e2, "MSE entre $a_{i,exacto}$ y $a_{i}$", 
#                 "{}".format(porcentaje),"ejer_1_mse_a_porcentaje.pdf",i)        

#     ejemplos_vector = [20,40,60,80,100,120]
#     i=-1
#     for ejemplos in ejemplos_vector:
#         i+=1
#         e1, e2= [], []
#         aux1, aux2= 0,0
#         for index in range(len(dimensiones)):
#             if dimensiones[index]> ejemplos: break
#             aux1, aux2 = regresion_lineal_numpy(dimensiones[index], ejemplos)
#             e1.append(aux1)
#             e2.append(aux2)
        
#         plot_me(3, dimensiones, e1, "MSE entre $y_{exacto}$ e $y_{esperado}$", 
#                 "{}".format(ejemplos), "ejer_1_mse_y_ejemplos.pdf", i)

#         plot_me(4, dimensiones, e2, "MSE entre $a_{i,exacto}$ y $a_{i}$", 
#                 "{}".format(ejemplos),"ejer_1_mse_a_ejemplos.pdf", i)   

#     plt.show()

def ejer1():
    ejemplos = np.arange(110,300,1)

    e1, e2= [],[]
    e3, e4= [],[]
    e5, e6= [],[]
    e7, e8= [],[]


    for ejemplo in ejemplos:
        #i+=1

        aux1, aux2 = regresion_lineal_numpy(40, ejemplo)
        e1.append(aux1)
        e2.append(aux2)

        aux1, aux2 = regresion_lineal_numpy(60, ejemplo)
        e3.append(aux1)
        e4.append(aux2)


        aux1, aux2 = regresion_lineal_numpy(80, ejemplo)
        e5.append(aux1)
        e6.append(aux2)

        aux1, aux2 = regresion_lineal_numpy(100, ejemplo)
        e7.append(aux1)
        e8.append(aux2)


    plt.figure(1)
    plt.title("MSE entre $y_{exacto}$ e $y_{esperado}$")
    plt.plot(ejemplos,e1, label="d=40", c=cmap(1))
    plt.plot(ejemplos,e3, label="d=60", c=cmap(2))
    plt.plot(ejemplos,e5, label="d=80", c=cmap(4))
    plt.plot(ejemplos,e7, label="d=99", c=cmap(5))    
    plt.ylabel("MSE")
    plt.xlabel("Ejemplos")
    plt.legend(loc=0)

    plt.figure(2)
    plt.title("MSE entre $a_{exacto}$ e $a_{esperado}$")
    plt.plot(ejemplos,e2, label="d=40",c=cmap(6))
    plt.plot(ejemplos,e4, label="d=60",c=cmap(7))
    plt.plot(ejemplos,e6, label="d=80",c=cmap(8))
    plt.plot(ejemplos,e8, label="d=99",c=cmap(9))
    plt.ylabel("MSE")
    plt.xlabel("Ejemplos")
    plt.legend(loc=0)

    plt.show()
    #plot_me(3, ejemplos, e1, "MSE entre $y_{exacto}$ e $y_{esperado}$", 
    #             "d=50", "ejer_1_mse_y_ejemplos.pdf", i)


    pass

def main( ):
    ejer1()
    pass

if __name__ == '__main__':
    main()