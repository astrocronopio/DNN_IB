import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,  
	'figure.figsize': [13, 8],  
	'figure.autolayout': True, 
	'font.family': 'serif', 
	'font.sans-serif': ['Times']}) 



class functor(object):
	"""docstring for functor"""
	def __init__(self, minV, maxV):
		self.minV=minV
		self.maxV=maxV

	def __call__(self):
		return  np.random.uniform(self.minV, self.maxV) 
		


class Noiser(functor):
	"""docstring for Noiser"""

	def __init__(self,minV, maxV):
		 super().__init__(minV, maxV)

	def __call__(self, x):
		return  x + super().__call__()


	

def ejer15():
	minV=-0.2
	maxV=0.2
	noiser= Noiser(minV,maxV)
	make_noise=np.vectorize(noiser)

	x = np.linspace(0, 10, 100)

	plt.xlabel("Tiempo [u.a.]")
	plt.ylabel(u"Señal [u.a.]")
	plt.plot(x,np.sin(x), color='black', ls='--', alpha=0.7, label="Señal")
	plt.plot(x,make_noise(np.sin(x)), lw=2, color='red', alpha=0.8, label="Señal + Noiser[{:.2},{:.2}]".format(minV, maxV))
	plt.legend(loc=0)
	plt.show()
	

	print(noiser(3))

	pass

def main():
	ejer15()
	pass


if __name__== "__main__":
	main()
