#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
Codigo para la practica 0 de la materia de Deep Learning del Instituto Balseiro
Escrito por Evelyn  Coronel
Agosto 2020
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
from ejer.ejer6 import ejer6
###############################################################
from ejer.ejer7 import ejer7
###############################################################
from ejer.geometria import ejer8
############################################################
import p0_lib
from p0_lib import rectangulo
from p0_lib.circunferencia import PI, area
from p0_lib.elipse import area
from p0_lib.rectangulo import area as area_rect

def ejer9():
	print(u"Librería p0_lib importada con éxito.")
	pass
############################################################
from ejer.ejer10 import ejer10
############################################################
from ejer.ejer11 import ejer11
############################################################
from ejer.ejer12 import ejer12
############################################################
from ejer.ejer13 import ejer13
###############################################################
from ejer.ejer14 import ejer14
###############################################################
from ejer.ejer15 import ejer15



def main():
	pract_0={	1 : ejer1,	2 : ejer2,	3 : ejer3,	4 : ejer4,
				5 : ejer5,	6 : ejer6,	7 : ejer7,  8 : ejer8,
				9 : ejer9,	10: ejer10,	11: ejer11, 12: ejer12,
				13: ejer13, 14: ejer14,	15: ejer15 			}
	
	while True:
		try:
			n = int(input(u"Ingrese el número de ejercicio que quiere ejecutar: "))
			if  n>15 or n<0: 
				print("Ingrese un número válido menor a 15")
				continue	
			break
		except ValueError:
			print("Ingrese un número válido menor a 15")
		
	pract_0[n]()


if __name__== "__main__":
	main()
