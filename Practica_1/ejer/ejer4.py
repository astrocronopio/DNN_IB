from ejer.ejer3 import knearest, knn_implementacion
from ejer.ejer2 import cluster_generator, kmean

import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})


def knn_custom_cluster(n_cluster=5, k=1):
    vectores_per_cluster=100
    vectores_cluster=vectores_per_cluster*n_cluster

    cluster, p_vect = cluster_generator(n_cluster,2,vectores_per_cluster)
    
    knn=kmean(100,n_cluster)
    knn.clasificar(cluster, p_vect)

    (x_train, y_train) = (cluster[:int(0.75*vectores_cluster)], knn.cluster_label[:int(0.75*vectores_cluster)])
    (x_test, y_test) = (cluster[int(0.75*vectores_cluster):], knn.cluster_label[int(0.75*vectores_cluster):])
    
    data=(x_train, y_train), (x_test, y_test)
    model_knn = knn_implementacion(data,len(y_test),k)
    

    steps=200
    xmin, xmax = cluster[:,0].min() - 1, cluster[:,0].max() + 1
    ymin, ymax = cluster[:,1].min() - 1, cluster[:,1].max() + 1
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)
    # Make predictions across region of interest
    labels = model_knn.predecir(np.c_[xx.ravel(), yy.ravel()])
    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)
    plt.contourf(xx,yy,z, cmap='rainbow', alpha=0.2)
    #plt.contour(xx, yy, z , 2, colors='black')

    knn.print_cluster_label(cluster, p_vect)
    plt.title("Número de vecinos k={}".format(k))

    plt.show()
    pass

def ejer4():
    print("k=1")
    knn_custom_cluster(5,1)
    exit()
    print("k=3")
    knn_custom_cluster(5,3)
    print("k=7")
    knn_custom_cluster(5,7)

def main():
    ejer4()
    pass

if __name__ == '__main__':
    main()
    