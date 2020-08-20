import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 24,  
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
	noiser= Noiser(-0.1,0.1)
	make_noise=np.vectorize(noiser)

	x = np.linspace(-4, 4, 100)

	plt.plot(x,make_noise(np.sin(x)))
	plt.show()
	

	print(noiser(3))

	pass

def main():
	ejer15()
	pass


if __name__== "__main__":
	main()
