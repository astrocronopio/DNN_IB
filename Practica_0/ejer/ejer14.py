import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 24,  
	'figure.figsize': [13, 8],  
	'figure.autolayout': True, 
	'font.family': 'serif', 
	'font.sans-serif': ['Times']}) 

num_grupos=1000

def hay_cumple_en_el(grupo):
	
	for i in range(len(grupo)):
		for j in range(len(grupo)):
			if grupo[i]==grupo[j] and i!=j:
				return 1
	return 0

def feliz_cumpleannos(personas, num_personas):
	fiesta=0
	for grupo in personas:
		fiesta = fiesta + hay_cumple_en_el(grupo)

	return  fiesta

def tabla_pers_porc(num_personas):
	
	personas = np.random.randint(1,365, (num_grupos,num_personas)) #366 limite sup porque randint lo excluye
	porcentaje = feliz_cumpleannos(personas, num_personas)
	print(num_personas, porcentaje)
	return  100*porcentaje/num_grupos


def ejer14():

	num_personas_vector = [10,20,30,40,50,60]
	num_probabilidad_vec= []
	cell_text=[]
	
	for num_personas in num_personas_vector:
		num_probabilidad_vec.append(tabla_pers_porc(num_personas))

	# for p,pr in zip(num_personas_vector, num_probabilidad_vec) :
	#     cell_text.append([p,pr])
	
	# cell_text.reverse()

	# columns = ('Personas', 'Probabilidad')

	# the_table = plt.table(cellText=cell_text,
 #                      colLabels=columns,
 #                      loc='right')
	# the_table.scale(1, 1.5)

	plt.ylabel("Probabilidad [%]")
	plt.xlabel("Personas por grupo")
	plt.plot(num_personas_vector, num_probabilidad_vec, color='red', alpha=0.6)
	plt.scatter(num_personas_vector, num_probabilidad_vec, color='red', alpha=0.6)
	plt.show()

def main():
	ejer14()
	pass


if __name__== "__main__":
	main()
