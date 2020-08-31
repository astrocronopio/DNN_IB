import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 24,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Times']})

color= ['red', 'blue', 'black', 'green', 'pink']

###################################################################
class kmean(object):
    centroides = []
    cluster_label=[]
    def __init__(self, total_iteraciones, n_cluster):
        self.total_iteraciones=total_iteraciones
        self.n_cluster=n_cluster
        pass

    def inicializar_centroides(self, cluster):
        index_random= np.random.permutation(cluster)
        return index_random[:self.n_cluster]

    def encontrar_centroides(self, cluster):
        centroides = np.zeros((self.n_cluster, cluster.shape[1])) # la dim que usamos
        
        for k in range(self.n_cluster):
            centroides[k] = np.mean( cluster[self.cluster_label==k,:], axis=0)
        
        return centroides
    
    def encontrar_cluster(self, distancia):
        return np.argmin(distancia,  axis=1)

    def calcular_distancia(self, cluster, centroids): 
        distancia = np.zeros((cluster.shape[0], self.n_cluster))
        for k in range(self.n_cluster):
            distancia[:,k] = np.square(np.linalg.norm( cluster - centroids[k], axis=1)) 
        return  distancia

    def print_cluster_label(self,cluster, p_vect):
        for k in range(self.n_cluster):
            aux= cluster[self.cluster_label==k,:]
            plt.scatter(aux[:,0], aux[:,1], color=color[k])
            plt.scatter(self.centroides[k,0], self.centroides[k,1], s=80)
            plt.scatter(p_vect[k,0], p_vect[k,1], s=80, marker='*', color='black')
                
        plt.pause(0.5)

    def clasificar(self, cluster, p_vect):
        self.centroides=self.inicializar_centroides(cluster)
        self.cluster_label= np.random.randint(0,self.n_cluster-1, size=cluster.shape[0])

        for _ in range(self.total_iteraciones):
            old_centroides      = self.centroides
            distancia           = self.calcular_distancia(cluster, old_centroides)
            self.cluster_label  = self.encontrar_cluster(distancia)
            self.centroides     = self.encontrar_centroides(cluster)

            self.print_cluster_label(cluster, p_vect)  

            if np.all(old_centroides == self.centroides): #aca se sale porque coverge
                break 
###################################################################
#p_vect=np.random.uniform(-5,5,size=(p,n))
def cluster_generator(p,n,N):
    p_vect = np.random.uniform(-5,5,size=(p,n))
    cluster=np.zeros((p*N,n))

    for index in range(p):         
        cluster[index*N: index*N + N] = np.random.normal(p_vect[index], 0.5, size=(N,n))

    return cluster, p_vect 

def kmeans_core(p=5, n=2, N=70):
    n_clusters=5
    cluster, p_vect = cluster_generator(p,  n, N) 
    plt.scatter(cluster[:,0], cluster[:,1])
    plt.pause(2)
    tkm = kmean(100, n_clusters)
    tkm.clasificar(cluster, p_vect)

def ejer2():
    kmeans_core()
    pass

def main():
    ejer2()
    pass


if __name__ == "__main__":
    main()
    plt.pause(10)
    pass