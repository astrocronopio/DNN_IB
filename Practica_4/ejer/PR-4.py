import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.optimizers import SGD
import keras as keras
import matplotlib.pyplot as plt
import random as random
import matplotlib as mpl
mpl.rcParams.update({'font.size': 19,  'figure.figsize': [8, 6],  'figure.autolayout': True, 'font.family': 'serif', 'font.sans-serif': ['Helvetica']})

colormap = plt.cm.gist_ncar


def ejer_1_arch_2_2_1():
	training_data = np.array([[1,1],[1,-1],	[-1,1],	[-1,-1]])
	target_data = 	np.array([[1],	[-1],	[-1],	[1]])

	fig1, ax1 = plt.subplots()
	fig2, ax2 = plt.subplots()

	ax1.set_title('Exactitud del modelo 2-2-1')
	ax1.set_xlabel('Épocas')
	ax1.set_ylabel('Exactitud')

	ax2.set_title('Pérdida del modelo 2-2-1')
	ax2.set_xlabel('Épocas')
	ax2.set_ylabel('Pérdida')

	plot_num=8
	fig1.gca().set_prop_cycle(plt.cycler('color', plt.cm.gnuplot(np.linspace(0, 1, plot_num))))
	fig2.gca().set_prop_cycle(plt.cycler('color', plt.cm.gnuplot(np.linspace(0, 1, plot_num))))

	for x in range(plot_num):
		model = Sequential()
		model.add(Dense(2, input_dim=2, activation='tanh', use_bias=True, bias_initializer='random_uniform'))
		model.add(Dense(1, activation='tanh', use_bias=True, bias_initializer='random_uniform'))
	
		sgd = SGD(lr=0.05)
	
		model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
		
		history = model.fit(training_data, target_data, batch_size=1, epochs=250, verbose=0)
		ax1.plot(history.history['accuracy'], label=str(x+1))
		ax2.plot(history.history['loss'], label=str(x+1)	)
		
		print("Para el caso {0}".format(x))	
		print(model.predict(training_data))

	ax2.legend(loc='upper right', title="Trials", ncol=2)
	ax1.legend(loc='lower right', title="Trials", ncol=2)

	fig1.savefig('ejer_1_2-2-1_acc.png')
	fig2.savefig('ejer_1_2-2-1_los.png')
	pass


def ejer_1_arch_2_1_1():
	training_data = np.array([[1,1],[1,-1],	[-1,1],	[-1,-1]])
	target_data = 	np.array([[1],	[-1],	[-1],	[1]])

	fig1, ax1 = plt.subplots()
	fig2, ax2 = plt.subplots()

	ax1.set_title('Exactitud del modelo 2-1-1')
	ax1.set_xlabel('Épocas')
	ax1.set_ylabel('Exactitud')

	ax2.set_title('Pérdida del modelo 2-1-1')
	ax2.set_xlabel('Épocas')
	ax2.set_ylabel('Pérdida')

	plot_num=8
	fig1.gca().set_prop_cycle(plt.cycler('color', plt.cm.gnuplot(np.linspace(0, 1, plot_num))))
	fig2.gca().set_prop_cycle(plt.cycler('color', plt.cm.gnuplot(np.linspace(0, 1, plot_num))))

	for y in range(plot_num):

		inp1 = keras.layers.Input(shape=(2,))
		#inp2 = keras.layers.Input(shape=(1,))
		
		#x = keras.layers.Concatenate()([inp1, inp2])
		x = keras.layers.Dense(1, activation='tanh', use_bias=True, bias_initializer='random_uniform')(inp1)
		
		x = keras.layers.Concatenate()([inp1, x])
		
		output = keras.layers.Dense(1, activation='tanh', use_bias=True, bias_initializer='random_uniform')(x)
		
		model = keras.models.Model(inputs=inp1, outputs=output) 
		#model.summary()
		sgd = SGD(lr=0.05)
	
		model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
		
		history = model.fit(training_data, target_data, batch_size=1, epochs=250, verbose=0)
		ax1.plot(history.history['accuracy'], label=str(y+1))
		ax2.plot(history.history['loss'], label=str(y+1)	)
		
		print("Para el caso {0}".format(x))	
		print(model.predict(training_data))

	ax2.legend(loc='upper right', title="Trials", ncol=4, labelspacing=0.2)
	ax1.legend(loc='lower right', title="Trials", ncol=2, labelspacing=0.2)

	fig1.savefig('ejer_1_2-1-1_acc.png')
	fig2.savefig('ejer_1_2-1-1_los.png')
	pass



def XOR_training(N, N_train):
	training_data	=  np.ones(N*N_train).reshape(N_train,N)
	target_data		=  np.ones(N_train).reshape(N_train,1)


	for y in range(N_train):
		product=1
		aux=1

		for x in range(N):
			aux= 1 if random.random() < 0.5 else -1
			product=product*aux
			training_data[y][x]=aux

		target_data[y][0]=product	

	return training_data, target_data

def ejer_2_N_Nprime_XOR(N, N_prime, N_train):

	training_data, target_data = XOR_training(N, N_train)

	model = Sequential()
	model.add(Dense(N_prime, input_dim=N, activation='tanh', use_bias=True, bias_initializer='random_uniform'))
	model.add(Dense(1, activation='tanh', use_bias=True, bias_initializer='random_uniform'))
	
	sgd = SGD(lr=0.05)
	
	model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
	
	history = model.fit(training_data, target_data, batch_size=1, epochs=250, verbose=0)

	return history.history['accuracy'], history.history['loss']

def ejer_2():
	fig1, ax1 = plt.subplots()
	fig2, ax2 = plt.subplots()

	plot_num=8
	fig1.gca().set_prop_cycle(plt.cycler('color', plt.cm.gnuplot(np.linspace(0, 1, plot_num))))
	fig2.gca().set_prop_cycle(plt.cycler('color', plt.cm.gnuplot(np.linspace(0, 1, plot_num))))

	ax1.set_title('Exactitud en el modelo $N$, $N\'$, $N_{train}$')
	ax1.set_xlabel('Épocas')
	ax1.set_ylabel('Exactitud')

	ax2.set_title('Pérdida en el modelo $N$, $N\'$, $N_{train}$')
	ax2.set_xlabel('Épocas')
	ax2.set_ylabel('Pérdida')


	structure=[	[4,4,5],
				[4,4,8],
				[4,4,16],
				[4,7,16],
				[4,2,16],
				[7,7,50],
				[7,7,100],
				[7,21,100]]

	ac1, er1 = ejer_2_N_Nprime_XOR(4,4,5)
	ax1.plot(ac1, label="(4,4,5)")
	ax2.plot(er1, label="(4,4,5)")

	ac2, er2 = ejer_2_N_Nprime_XOR(4,4,8)
	ax1.plot(ac2, label="(4,4,8)")
	ax2.plot(er2, label="(4,4,8)")

	ac3, er3 = ejer_2_N_Nprime_XOR(4,4,16)
	ax1.plot(ac3, label="(4,4,16)")
	ax2.plot(er3, label="(4,4,16)")

	ac4, er4 = ejer_2_N_Nprime_XOR(4,3,16)
	ax1.plot(ac4, label="(4,3,16)")
	ax2.plot(er4, label="(4,3,16)")

	ac5, er5 = ejer_2_N_Nprime_XOR(4,1,16)
	ax1.plot(ac5, label="(4,1,16)")
	ax2.plot(er5, label="(4,1,16)")

	ac6, er6 = ejer_2_N_Nprime_XOR(7,10,100)
	ax1.plot(ac6, label="(7,10,100)")
	ax2.plot(er6, label="(7,10,100)")

	ac7, er7 = ejer_2_N_Nprime_XOR(7,7,100)
	ax1.plot(ac7, label="(7,7,100)")
	ax2.plot(er7, label="(7,7,100)")

	ac8, er8 = ejer_2_N_Nprime_XOR(7,4,100)
	ax1.plot(ac8, label="(7,4,100)")
	ax2.plot(er8, label="(7,4,100)")


	ax2.legend(loc='upper right', title='$N$, $N\'$, $N_{train}$', ncol=2)
	ax1.legend(loc='lower right', title='$N$, $N\'$, $N_{train}$', ncol=2)

	fig1.savefig('ejer_2_acc.png')
	fig2.savefig('ejer_2_los.png')
	pass

def logictic_function(x):
	return 4*x*(1-x)

def logistic_training(N, epochs_total):
	training_data	=  np.ones(N).reshape(N,1)
	target_data		=  np.ones(N).reshape(N,1)

	for x in range(N):
		training_data[x][0]=random.uniform(0,1)
		target_data[x][0]=logictic_function(target_data[x][0])
		#for y in range(epochs_total):
		#	target_data[x][0]= logictic_function(target_data[x][0])

	return training_data, target_data

def ejer_3():
	fig1, ax1 = plt.subplots()
	fig2, ax2 = plt.subplots()

	ax1.set_title('Exactitud')
	ax1.set_xlabel('Épocas')
	ax1.set_ylabel('Exactitud')

	ax2.set_title(' Entrenamiento (-), Generalización ($\\cdot\\cdot$)')
	ax2.set_xlabel('Épocas')
	ax2.set_ylabel('Error')
	
	plot_num=6
	fig1.gca().set_prop_cycle(plt.cycler('color', plt.cm.gnuplot(np.linspace(0, 1, plot_num))))
	fig2.gca().set_prop_cycle(plt.cycler('color', plt.cm.gnuplot(np.linspace(0, 1, plot_num))))

	epochs_total=250
	training_data, target_data = logistic_training(115, epochs_total)

	ejemplos_array=[5,10,100]
	ejemplos_color=['red', 'black', 'blue']

	for y in range(len(ejemplos_color)):
		inp1 = keras.layers.Input(shape=(1,))
		x = keras.layers.Dense(5, activation='sigmoid', use_bias=True, bias_initializer='random_uniform')(inp1)
		#x = keras.layers.Dense(5, activation='sigmoid')(inp1)
		
		x = keras.layers.Concatenate()([inp1, x])
		
		output = keras.layers.Dense(1, activation='linear', use_bias=True, bias_initializer='random_uniform')(x)
		#output = keras.layers.Dense(1, activation='linear')(x)
		
		model = keras.models.Model(inputs=inp1, outputs=output) 
		#model.summary()
		sgd = SGD(lr=0.05,momentum=0.05)
		print(y)
		model.compile( optimizer=sgd, loss='MSE', metrics=['mae'])
		history = model.fit(training_data[:ejemplos_array[y]], target_data[:ejemplos_array[y]], validation_data=(training_data[ejemplos_array[y]:ejemplos_array[y]+14], target_data[ejemplos_array[y]:ejemplos_array[y]+14]), batch_size=14,shuffle=True, epochs=epochs_total, verbose=0)

		#ax1.plot(history.history['accuracy'])
		ax2.plot(history.history['val_mae'], ':', label=str(ejemplos_array[y]), color=ejemplos_color[y])
		ax2.plot(history.history['mae'], '-', label=str(ejemplos_array[y]), color=ejemplos_color[y])

	ax2.legend(loc='upper right', title="Ejemplos.", ncol=3)
	#ax1.legend(loc='lower right', title="Ejemplos", ncol=3)

	#fig1.savefig('ejer_3_acc.png')
	fig2.savefig('ejer_3_los_gen.png')
	pass
#******************************************************************************

def main():
	#ejer_1_arch_2_2_1()
	#ejer_1_arch_2_1_1()
	#ejer_2()
	#ejer_3()


if __name__== "__main__":
	main()
