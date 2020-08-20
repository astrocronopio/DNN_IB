#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from ejer.ejer8 import ejer8
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



def main():
	pract_0={	1 : ejer1,	2 : ejer2,
				3 : ejer3,	4 : ejer4,
				5 : ejer5,	6 : ejer6,
				7 : ejer7,  8 : ejer8,
				9 : ejer9,	10: ejer10,
				11: ejer11, 12: ejer12
			}

	n = int(input(u"Ingrese el número de ejercicio que quiere ejecutar: "))

	pract_0[n]()

	pass


if __name__== "__main__":
	main()
