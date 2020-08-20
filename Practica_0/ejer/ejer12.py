import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 24,  
	'figure.figsize': [12, 8],  
	'figure.autolayout': True, 
	'font.family': 'serif', 
	'font.sans-serif': ['Times']}) 


def f(x, y):
	return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
	

def ejer12():
	n = 1024
	X = np.random.normal(0,1,n)
	Y = np.random.normal(0,1,n)
	color_func = np.arctan2(Y, X)
	
	plt.ylim(-1.5,1.5)
	plt.xlim(-1.5,1.5)
	
	plt.scatter(X,Y, s=70, c=color_func, alpha=.6, cmap='jet', edgecolors='black')
	plt.xticks([])
	plt.yticks([])

	plt.show()

def main():
	ejer12()
	pass

if __name__ == '__main__':
	main()