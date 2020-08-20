import numpy as np
import matplotlib.pyplot as plt

#plt.axis([0, 10, 0, 1])

for i in range(10):
	plt.clf()
	y = np.random.random()
	x = np.random.random()
	plt.quiver(x, y)
	plt.pause(0.5)

plt.show()