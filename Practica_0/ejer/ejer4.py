from ejer.ejer3  import parametros,cuadratica
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 24,  
	'figure.figsize': [12, 8],  
	'figure.autolayout': True, 
	'font.family': 'serif', 
	'font.sans-serif': ['Times']}) 

def plot_parabola(a,b,c):
	soluciones = cuadratica(a,b,c)

	x = np.linspace(-4.5, 4, 100)
	y = a*x*x + b*x +c

	plt.xlabel("x")
	plt.ylabel("f(x)")
	plt.plot(x,np.zeros(len(x)), color='black', linestyle= '--', alpha=0.5)
	plt.plot(x,y, color='blue', alpha=0.8)

	try:
		if soluciones==None:
			plt.show()
	except ValueError:
		plt.scatter(soluciones, [0,0], color='red', alpha=0.8)

		plt.annotate("",xy=(soluciones[0],0), xytext=(soluciones[0],a*7),
					arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
		
		plt.text(soluciones[0],a*7, u"$x_1$={:.3}".format(soluciones[0]),
					         {'color': 'black', 'ha': 'center', 'va': 'center',
 					         'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})



		plt.annotate("", xy=(soluciones[1],0), xytext=(soluciones[1],a*5),
					arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
		
		plt.text(soluciones[1],a*5, u"$x_2$={:.3}".format(soluciones[1]),
					         {'color': 'black', 'ha': 'center', 'va': 'center',
 					         'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})
		plt.show()

def ejer4():
	print("Ejercicio 4\n")

	a,b,c = parametros()
	plot_parabola(a,b,c)

	pass

def main():
	ejer4()
	pass


if __name__== "__main__":
	main()
