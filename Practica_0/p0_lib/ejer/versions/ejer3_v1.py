import numpy as np

class layer(object):
    def __init__(self, input_size=16, output_size=1,  bias=False, activation=None, name='No name'):
        """
        La función de activación ya está necesariamente vectorizada
        y tiene que se ser inicializa o  sino  muere
        """
        self.input_size= input_size
        self.output_size=output_size
        self.bias= bias
        self.activation=activation
        self.name=name
        self.weight_init()
        if activation==None:
            print("Dame la función de  activacion\n")
            exit()

    def weight_init(self, initializer= np.random.uniform ):
        self.w = initializer(-1,1,shape=(self.input_size, self.output_size))
        
        if self.bias==True:
            self.b = initializer(-1,1)    
        elif self.bias==False:
            self.b = 0
        
    def local_gradient(self):
        pass

    def call(self, inputs):
        return self.activation(np.matmul(inputs, weight_init) + self.b)  
        

class DenseNetwork(object):
    def __init__(self, epochs=100, eta=0.01, lambda_L2=0.0):
        self.epochs=epochs
        self.eta=eta
        self.lambda_L2=lambda_L2
            

def ejer3():


    pass

def main():
    ejer3()
    pass


if __name__ == '__main__':
    main()
    




        