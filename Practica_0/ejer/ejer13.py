import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({	'font.size': 18, 'figure.figsize': [10, 8], 'figure.autolayout': True, 	'font.family': 'serif', 	'font.sans-serif': ['Times']}) 


class R2(object):
	"""docstring for R2"""
	def __init__(self, x=0,y=0):
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
			pos, vel = R2(), R2()
			pos.ini(L)
			while True:	 
				vel.ini(maxVel)
				if(vel.mod() < maxVel): break
			
			self.pez.append(Pez(pos, vel))

	def rule_1_2_3(self, pez, x, rc, vc):
		v1, v2, v3	= R2(), R2(), R2()

		v1 	= (rc - pez.pos)/8.0
		v3	= (vc - pez.vel)/8.0
			
		for j in range(self.n):
			if j==x: break
			delta= pez.pos - self.pez[j].pos
			if delta.mod()< maxDist: v2 = v2 + delta/delta.mod()

		return v1+ v2+ v3

		
	def doStep(self, boolean):
		rc, vc, aux, pos, vel = R2(), R2(), R2(), R2(), R2()
		vel_vec=[]
		#Calculo el rc y vc
		for x in range(self.n):
			pos = pos + self.pez[x].pos
			vel = vel + self.pez[x].vel
		
		rc, vc = pos/self.n, vel/self.n

		for x in range(self.n):
			aux= self.rule_1_2_3(self.pez[x], x, rc, vc)
			vel_vec.append(aux)
		
		for x in range(self.n):
			self.pez[x].vel =  self.pez[x].vel + vel_vec[x] 		# CAMBIO DE VELOCIDAD
			self.pez[x].pos =  self.pez[x].pos + self.pez[x].vel*dt # CAMBIO DE POSICION
			
			#Ahora verifico que esten acotadas las velocidades
			if self.pez[x].vel.mod() > maxVel:
				self.pez[x].vel = self.pez[x].vel*maxVel/self.pez[x].vel.mod()
			
			#Por si es que hice la corrida sin/con paredes
			if boolean==True:
				if abs(self.pez[x].pos.x)>L or abs(self.pez[x].pos.y)>L:
					self.pez[x].vel = self.pez[x].vel*(-1)


	def print(self, i, niter, boolean):
		plt.ylim(-L, L)
		plt.xlim(-L, L)
		ax = plt.axes()
		ax.set_facecolor('xkcd:sky blue')
		
		pos=R2()

		if boolean:
			plt.title("Iteración {} de {} ({:.3}s/{}s), con paredes duras".format(i+1,niter, (i+1)*pause, niter*pause))
		else:
			plt.title("Iteración {} de {} ({:.3}s/{}s), sin paredes".format(i+1,niter, (i+1)*pause, niter*pause))
			for x in range(self.n):
				pos = pos + self.pez[x].pos
			pos = pos/self.n
			if abs(pos.x)>L or abs(pos.y)>L:
				plt.ylim(-L+pos.y, L+pos.y)
				plt.xlim(-L+pos.x, L+pos.x)
				plt.title("Iteración {} de {} ({:.2}s/{}s), sin paredes\nCambiamos la ventana".format(i+1,niter, (i+1)*pause, niter*pause))
			

		for x in range(self.n):
			plt.quiver(self.pez[x].pos.x, self.pez[x].pos.y, self.pez[x].vel.x, self.pez[x].vel.y,  color='black', alpha=0.6, pivot='mid')
		
		plt.pause(0.5*pause)
		if i==0 or i==(niter-1) or i==int(niter*0.5):
			try:
				plt.savefig("docs/ejer_13_{}{}.pdf".format(i, boolean))
			except FileNotFoundError:
				plt.savefig("../docs/ejer_13_{}{}.pdf".format(i, boolean))
		plt.clf()



L, N, V , dt = 20, 16, 10, 0.1
maxVel = 4
maxDist= 3
pause= 0.08

def ejer13():
	while True:
		try:
			niter= int(input("Tarda {:.2} s por cada 10 iter, recomiendo 100. Choose wisely \n ¿Cuántas iteraciones quiere ver? \n".format(10*pause)))
			break
		except ValueError:
			print("Ingrese un número válido")

	boolean= False if int(input("¿Ponemos las paredes duras? Si (1)/ No (0) \n"))==0 else True
	mpl.patches.Rectangle([-L,-L], L, L)

	c = Cardumen()
	c.initialize(N, maxVel, maxDist)
	for i in range(niter):
		c.doStep(boolean)
		c.print(i, niter, boolean)
	plt.pause(0.1)
	exit()

def main():
	ejer13()
	pass

if __name__ == '__main__':
	main()