import numpy  as np


class layer(object):
    def __init__(self,  input_size=16, 
                        output_size=1,  
                        bias=False, 
                        activation=None, 
                        name='No name'):
        """
        La función de activación ya está necesariamente vectorizada
        y tiene que se ser inicializa o  sino  muere
        """
        self.input_size = input_size
        self.output_size=output_size
        self.bias       = bias
        self.activation =activation
        self.name       =name
        self.weight_init()

        if activation==None:
            print("Dame la función de  activacion\n")
            exit()

    def weight_init(self, initializer= np.random.uniform ):
        self.w = initializer(-1,1,shape=(self.input_size, self.output_size))
        
        if self.bias==True:
            self.w = np.hstack(( (initializer(-1,1, shape=self.output_size), self.w))) 

    def local_gradient(self):
        pass
    
    def __call__(self, inputs):
        inputs_copy= np.copy(inputs)
        
        if self.bias==True:
            inputs_copy = np.hstack(( (np.ones((len(inputs_copy),1) )), inputs_copy)) 
        
        return self.activation(np.matmul(inputs_copy, weight_init) + self.b)  
   