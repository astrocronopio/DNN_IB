import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 24,  
	'figure.figsize': [12, 8],  
	'figure.autolayout': True, 
	'font.family': 'serif', 
	'font.sans-serif': ['Times']}) 



def ejer2():
	print("Ejercicio 2\n")

	data= np.random.gamma(3,2,1000) #dataset
	mean= np.mean(data)
	stddev= np.std(data)
	plt.ylabel("Cuentas")
	plt.xlabel("x")
	plt.hist(data, bins=30, color='red', alpha=0.6,	label="Media={:.3}, $\\sigma$={:.3}".format(mean, stddev))

	plt.legend(loc=0)
	plt.show()

def main():
	ejer2()
	pass


if __name__== "__main__":
	main()
