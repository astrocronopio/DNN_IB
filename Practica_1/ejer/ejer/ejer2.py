import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
	'font.size': 20,
	'figure.figsize': [12, 8],
	'figure.autolayout': True,
	'font.family': 'serif',
	'font.sans-serif': ['Palatino']})



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
        plt.xticks([])
        plt.yticks([])
        cmap = plt.get_cmap('rainbow',self.n_cluster)

        for k in range(self.n_cluster):
            aux= cluster[self.cluster_label==k,:]
            plt.scatter(aux[:,0], aux[:,1], color=cmap(k))

        plt.scatter(self.centroides[:,0], self.centroides[:,1], s=200, marker='^', color='black', label="Centroides")
        plt.scatter(p_vect[:,0], p_vect[:,1], s=350, marker='*', color='black', alpha=0.6, label="Medias Gaussianas")
        plt.legend(loc=0)        
        plt.pause(0.5)
        

    def clasificar(self, cluster, p_vect, boolean=False):
        self.centroides=self.inicializar_centroides(cluster)
        self.cluster_label= np.random.randint(0,self.n_cluster-1, size=cluster.shape[0])

        for _ in range(self.total_iteraciones):
            old_centroides      = self.centroides
            distancia           = self.calcular_distancia(cluster, old_centroides)
            self.cluster_label  = self.encontrar_cluster(distancia)
            self.centroides     = self.encontrar_centroides(cluster)

            if(boolean==True):
                plt.clf() 
                self.print_cluster_label(cluster, p_vect)  

            if np.all(old_centroides == self.centroides): #aca se sale porque coverge
                if (boolean == True): 
                    plt.title(u"La clasificación convergió")
                    plt.show()
                    exit() 
                else: break
###################################################################

def cluster_generator(p,n,N):
    p_vect = np.random.uniform(-3,3,size=(p,n))
    cluster=np.zeros((p*N,n))
    sigma_p= np.random.uniform(0.3, 1.3, size=p)

    for index in range(p):         
        cluster[index*N: index*N + N] = np.random.normal(p_vect[index], sigma_p[index], size=(N,n))

    return cluster, p_vect 

def kmeans_core(p=4, n=5, N=70):
    n_clusters=4
    if (n_clusters>p):
        print("Estás intentando sobre-clasificar el cluster ")
    cluster, p_vect = cluster_generator(p,  n, N) 
    plt.xticks([])
    plt.yticks([])
    plt.title("Los puntos a clasificar")
    
    plt.scatter(cluster[:,0], cluster[:,1], c='red', alpha=0.7)
    plt.pause(2)
    tkm = kmean(100, n_clusters)
    tkm.clasificar(cluster, p_vect, True)

def ejer2():
    kmeans_core()
    pass

def main():
    ejer2()
    pass

if __name__ == "__main__":
    main()
    plt.title("La clasificación terminó")
    plt.show()
    pass