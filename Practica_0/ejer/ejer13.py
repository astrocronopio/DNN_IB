import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({	'font.size': 24, 'figure.figsize': [13, 8], 'figure.autolayout': True, 	'font.family': 'serif', 	'font.sans-serif': ['Times']}) 


class R2(object):
	"""docstring for R2"""
	def __init__(self, x,y):
		self.x, self.y = x,  y

	def __add__(self, other):
		return R2(self.x + other.x, self.y + other.y)

	def __sub__(self, other):
		return R2(self.x - other.x, self.y - other.y)

	def __neg__(self, other):
		R2(-self.x,  -self.y)

	def ini(self, Lim):
		self.x, self.y = np.random.uniform(-Lim,Lim), np.random.uniform(-Lim,Lim)
	
	def __truediv__(self, other):
		return R2(self.x/other, self.y/other)

	def __mul__(self, other):
		return R2(self.x*other, self.y*other)

	def mod(self):
		return np.sqrt(self.x**2 + self.y**2)

class Pez(object):
	"""docstring for Pez"""
	def __init__(self, pos, vel):
		self.pos, self.vel = pos, vel
		
class Cardumen(object):
	"""docstring for Cardumen"""
	n, 	maxVel, maxDist= 0,0,0
	pez=[]

	def initialize(self, n, maxVel, maxDist):
		self.n=n

		for x in range(n):
			pos, vel = R2(0,0), R2(0,0)
			pos.ini(L)
			while True:	 
				vel.ini(maxVel)
				if(vel.mod() < maxVel):
					break
			self.pez.append(Pez(pos, vel))
		pass

	def rule_1_2_3(self, pez, x, rc, vc):
		v1, v2, v3	= R2(0,0), R2(0,0), R2(0,0)

		v1 	= (rc - pez.pos)/8.0
		v3	= (vc - pez.vel)/8.0
			
		for j in range(self.n):
			if j==x: break
			
			delta= pez.pos - self.pez[j].pos
		
			if delta.mod()< maxDist: v2 = v2 + delta/delta.mod()

		return v1+ v2+ v3

		
	def doStep(self, boolean):
		rc, vc, aux = R2(0,0), R2(0,0), R2(0,0)
		pos, vel = R2(0,0), R2(0,0)

		for x in range(self.n):
			pos = pos + self.pez[x].pos
			vel = vel + self.pez[x].vel
			pass
		
		rc, vc = pos/self.n, vel/self.n
		
		vel_vec=[]
	
		for x in range(self.n):
			aux= self.rule_1_2_3(self.pez[x], x, rc, vc)
			vel_vec.append(aux)
		
		for x in range(self.n):
			self.pez[x].vel =  self.pez[x].vel + vel_vec[x] # CAMBIO DE VELOCIDAD
			self.pez[x].pos =  self.pez[x].pos + self.pez[x].vel*dt #CAMBIO DE POSICION
			
			#Ahora verifico que esten acotadas las velocidades
			if self.pez[x].vel.mod() > maxVel:
				self.pez[x].vel = self.pez[x].vel*maxVel/self.pez[x].vel.mod()
		
			if boolean==True:
				if abs(self.pez[x].pos.x)>L or abs(self.pez[x].pos.y)>L:
					self.pez[x].vel = self.pez[x].vel*(-1)


	def print(self, i, niter, boolean):
		plt.ylim(-L, L)
		plt.xlim(-L, L)
		plt.xticks([])
		plt.yticks([])
		
		pause= 0.08
		if boolean:
			plt.title("Iteración {} de {} ({:.3}s/{}s), con paredes duras".format(i+1,niter, (i+1)*pause, niter*pause))
		else:
			plt.title("Iteración {} de {} ({:.3}s/{}s), sin paredes".format(i+1,niter, (i+1)*pause, niter*pause))

		for x in range(self.n):
			plt.quiver(self.pez[x].pos.x, self.pez[x].pos.y, self.pez[x].vel.x, self.pez[x].vel.y,  color='red', alpha=0.6)
		
		plt.pause(0.5*pause)
		plt.clf()



L, N, V , dt = 40, 16, 10, 0.5
maxVel = 4
maxDist= 3

def ejer13():
	
	niter= int(input("¿Cuántas iteraciones quiere ver? \n"))
	boolean= False if int(input("¿Ponemos las paredes duras? Si (1)/ No (0) \n"))==0 else True


	c = Cardumen()
	c.initialize(N, maxVel, maxDist)
	for i in range(niter):
		c.doStep(boolean)
		c.print(i, niter, boolean)
	plt.show()

def main():
	ejer13()
	pass

if __name__ == '__main__':
	main()