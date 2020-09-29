import numpy as np


class regularizador(object):
    def __init__(self, l):
        self.l = l

    def __call__(self, W):
        pass


class L1(regularizador):
    def __call__(self,W):
        return self.l * np.sum(np.abs(W))
    
    def derivate(self, W):
        return self.l*np.sign(W)

class L2(regularizador):
    def __call__(self,W):
        return self.l* np.sum(W*W) 

    def derivate(self, W): 
        return 2.0*self.l*W  