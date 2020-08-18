import numpy as np
import ejer.circunferencia as circle
import ejer.rectangulo as rectang


def ejer8():
	print("Ejercicio 8\n")

	a,b = 2,4
	print(u"Área rectángulo de {}x{} es {}".format(a,b,rectang.area(a,b)))

	r=4

	print(u"Área círculo de {} de radio es {}".format(r,circle.area(r)))

def main():
	ejer8()
	pass

if __name__ == '__main__':
	main()