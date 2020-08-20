import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 24,  
	'figure.figsize': [13, 8],  
	'figure.autolayout': True, 
	'font.family': 'serif', 
	'font.sans-serif': ['Times']}) 


def f(x, y):
	return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
	

def ejer11():
	n = 256
	x = np.linspace(-3, 3, 4 * n)
	y = np.linspace(-3, 3, 3 * n)
	X, Y = np.meshgrid(x,y)
	plt.contourf(X, Y, f(X, Y), 6, alpha=.75, cmap=plt.cm.hot)
	C = plt.contour(X, Y, f(X,Y), 6, colors='black')
	plt.clabel(C, inline=1, fontsize=10)
	plt.xticks([])
	plt.yticks([])


# Add a title an

	# plt.imshow(f(X, Y), cmap='bone')
	# #cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
	# plt.colorbar() 
	# plt.axis('off')
	plt.show()

def main():
	ejer11()
	pass

if __name__ == '__main__':
	main()