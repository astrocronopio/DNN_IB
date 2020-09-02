#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
Codigo para la practica 1 de la materia de Deep Learning del Instituto Balseiro
Escrito por Evelyn  Coronel
Septiembre 2020
'''

###############################################################
from ejer.ejer1 import ejer1
###############################################################
from ejer.ejer2 import ejer2
###############################################################
from ejer.ejer3 import ejer3
###############################################################
from ejer.ejer4 import ejer4
###############################################################
from ejer.ejer5 import ejer5
###############################################################
def main():
	pract_0={	1 : ejer1,	2 : ejer2,	3 : ejer3,	4 : ejer4, 5 : ejer5}
	
	while True:
		try:
			n = int(input(u"Ingrese el número de ejercicio que quiere ejecutar: "))
			if  n>5 or n<0: 
				print("Ingrese un número válido menor a 6")
				continue	
			break
		except ValueError:
			print("Ingrese un número válido menor a 6")
		
	pract_0[n]()


if __name__== "__main__":
	main()