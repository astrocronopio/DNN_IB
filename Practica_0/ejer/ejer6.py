import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 24,  
	'figure.figsize': [12, 8],  
	'figure.autolayout': True, 
	'font.family': 'serif', 
	'font.sans-serif': ['Times']}) 

from ejer.ejer5 import ejer5,Lineal
###############################################################

class Exponencial(Lineal):
	"""docstring for Exponencial"""
	def __init__(self, a, b):
		super(Exponencial, self).__init__(a,b)
	
	def solucion(self, c):
		super().solucion(c)
		logx=(np.log(c) - np.log(self.b))/np.log(self.a)
		return np.exp(logx)

	def f(self, x):
		super().f(x)
		return self.a*pow(x, self.b)

	def __str__(self):
		return "f(x) =  {}*x^({})".format(self.a,self.b)

def ejer6():
	print("Ejercicio 6: f(x) =  ax^b\n")

	a,b,x,y=2,3,4,4

	f = Exponencial(a,b)

	print(f)
	print(u"Solución a f(x)={} es  x={:0.5}".format(y, f.solucion(y)))
	print("Función evaluada en {}: {}".format(x, f.f(x)))


def main():
	ejer6()
	pass

if __name__ == '__main__':
	main()