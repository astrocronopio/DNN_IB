import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 24,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Times']})

color= ['red', 'blue', 'black', 'green']

###################################################################
class kmean(object):
    centroides = []
    cluster_label=[]
    def __init__(self, total_iteraciones, n_cluster):
        self.total_iteraciones=total_iteraciones
        #self.cluster=cluster
        self.n_cluster=n_cluster
        pass

    def inicializar_centroides(self, cluster):
        index_random= np.random.permutation(cluster)
        return index_random[:self.n_cluster]
        #return np.random.choice(cluster, self.n_cluster)

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

    def print_cluster_label(self,cluster):
        for k in range(self.n_cluster):
            aux= cluster[self.cluster_label==k,:]
            plt.scatter(aux[:,0], aux[:,1], color=color[k])    
        plt.pause(0.5)
        plt.clf()

    def clasificar(self, cluster):
        self.centroides=self.inicializar_centroides(cluster)
        self.cluster_label= np.random.randint(0,self.n_cluster-1, size=cluster.shape[0])

        for _ in range(self.total_iteraciones):
            old_centroides      = self.centroides
            distancia           = self.calcular_distancia(cluster, old_centroides)
            self.cluster_label  = self.encontrar_cluster(distancia)
            self.centroides     = self.encontrar_centroides(cluster)

            self.print_cluster_label(cluster)  

            if np.all(old_centroides == self.centroides): #aca se sale porque coverge
                break 
###################################################################

def cluster_generator(p,n,N):
    p_vect=np.random.uniform(-5,5,size=(p,n))
    cluster=np.zeros((p*N,n))

    for index in range(p):         
        cluster[index*N: index*N + N] = np.random.normal(p_vect[index], 0.3, size=(N,n))

    return cluster 
    #cluster= np.array(cluster)
    #return np.random.shuffle(cluster)
    
def kmeans_core(p=4, n=2, N=30):
    n_clusters=4
    cluster = cluster_generator(p,  n, N) 
    plt.scatter(cluster[:,0], cluster[:,1])
    plt.pause(2)
    tkm = kmean(100, n_clusters)
    tkm.clasificar(cluster)



def ejer2():
    kmeans_core()
    pass

def main():
    ejer2()
    pass


if __name__ == "__main__":
    main()
    pass