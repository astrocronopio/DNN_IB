#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
Codigo para la practica 2 de la materia de Deep Learning del Instituto Balseiro
Escrito por Evelyn  Coronel
Septiembre 2020
'''
  
######################################
from ejer.ejer1 import ejer1

def main():
	pract_2={	3 : ejer3,	4 : ejer4, 
				5 : ejer5,  6 : ejer6,	
				7 : ejer7,	8 : ejer8,	
				9 : ejer9	}
	
	while True:
		try:
			n = int(input(u"Ingrese el número de ejercicio que quiere ejecutar: "))
			if  n>9 or n<3: 
				print("Ingrese un número válido menor entre 3-9")
				continue	
			break
		except ValueError:
			print("Ingrese un número válido de ejercicio")
		
	pract_2[n]()


if __name__== "__main__":
	main()