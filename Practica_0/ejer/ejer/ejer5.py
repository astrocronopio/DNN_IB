#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 24,  
	'figure.figsize': [12, 8],  
	'figure.autolayout': True, 
	'font.family': 'serif', 
	'font.sans-serif': ['Times']}) 

class Lineal():
	"""docstring for Lineal"""
	def __init__(self, a, b):
		self.a = a
		self.b = b

	def solucion(self, c):
		if self.a==0:
			print(u"Solución indefinida")
		else:
			return (c -self.b)/self.a
	
	def f(self, x):
		return self.a*x + self.b

	def __str__(self):
		return "f(x) =  {}x + {}".format(self.a,self.b)

def ejer5():
	print("Ejercicio 5: f(x) =  ax +b\n")

	a,b,x,y=8,5,7,4

	f = Lineal(a,b)

	print(f)
	print(u"Solución a f(x)={} es  x={:0.5}".format(y, f.solucion(y)))
	print("Función evaluada en {}: {}".format(x, f.f(x)))

	pass

def main():
	ejer5()
	pass

if __name__ == '__main__':
	main()