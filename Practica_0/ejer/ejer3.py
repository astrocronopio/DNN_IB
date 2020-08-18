import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 24,  
	'figure.figsize': [12, 8],  
	'figure.autolayout': True, 
	'font.family': 'serif', 
	'font.sans-serif': ['Times']}) 

def cuadratica(a,b,c):
	delta = b*b - 4*a*c 
	
	if delta <0:
		print(" Las soluciones de ({})x^2 + ({})x + ({}) = 0 no son reales".format(a,b,c))
		return 
	else:
		x1= (-b + np.sqrt(delta))/(2*a)
		x2= (-b - np.sqrt(delta))/(2*a)
		return np.array([x1,x2])

def parametros():

	while True:
		try:
			a = float(input("a="))
			break
		except ValueError:
			print(u"El parámetro a es necesario.")
	try:
		b = float(input("b="))
	except ValueError:
		print("valor predeterminado para b=0")
		b=0
	
	try:
		c = float(input("c="))
	except ValueError:
		print("valor predeterminado para c=0")
		c=0
	
	return a,b,c

def ejer3():
	print("Ejercicio 3\n")

	a,b,c = parametros()
	
	solucion = cuadratica(a, b, c)
	
	try:
		if  solucion == 'NoneType':
			exit()
	except ValueError:
		print("Soluciones de ({})x^2 + ({})x + ({}) = 0:".format(a,b,c))
		print(u"(x_1, x_2) = {}".format(solucion))
	pass


def main():
	ejer3()
	pass


if __name__== "__main__":
	main()
