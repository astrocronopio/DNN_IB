import numpy as np


def ejer1():
	print("Ejercicio 1\n")
				# x	 y	z	
	matriz	= [	[ 1, 0, 1], 
				[ 2,-1, 1],
				[-3, 2,-2]]
	
	b		= [-2, 1, -1]

	if np.linalg.det(matriz)==0:
		print("Matriz no invertible")
	
	else:
		solucion= np.linalg.solve(matriz, b)
		print(u"La solución (x,y,z) es {}".format(solucion))
	pass

def main():
	ejer1()
	pass


if __name__== "__main__":
	main()
