import numpy as np

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})


x= np.linspace(-5,5,100)
y= np.sin(x)

plt.plot(x,y)
plt.savefig("pepe.png")
plt.close()