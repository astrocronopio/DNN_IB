import numpy as np
import ejer.circunferencia as circle

from ejer.circunferencia import area, PI

def ejer7():


	radio=4
	print("\nimport as circle")
	print("pi={}".format(circle.PI))
	print(u"Área: {} ".format(circle.area(radio)))


	print("\nfrom ... import ")
	print("pi={}".format(PI))
	print(u"Área para radio = {}: {} ".format(radio, area(radio)))

	print("¿circle.PI y PI son el mismo objeto?")
	print(circle.PI is PI)

	print("¿circle.area y area son el mismo objeto?")
	print(circle.area is area)


	pass

def main():
	ejer7()
	pass


if __name__== "__main__":
	main()
