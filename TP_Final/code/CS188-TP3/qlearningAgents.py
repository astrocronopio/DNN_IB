# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import numpy as np

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        """
        ###############################################
        ###############################################
        ###############################################
        """
        self.values  = util.Counter() 
        #Counter es un dictonario de accion/states
        
        #Ya esta implementado por ellos, 
        #Usan en todas partes los metodos de esta clase, 
        #entonces es esta  la respuesta
        """
        ###############################################
        ###############################################
        ###############################################
        """

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        """
        ###############################################
        ###############################################
        ###############################################
        """
        return self.values[(state,action)] #El par estado, action para el q-val
        """
        ###############################################
        ###############################################
        ###############################################
        """
        util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        """
        ###############################################
        ###############################################
        ###############################################
        """
        
        q_val = -np.inf #lo inicio lo más negativo posible para no
        #confundir al algoritmo

        for action in self.getLegalActions(state):
            q_val  = max(self.getQValue(state, action), q_val) #Q-magic XD
      
        #EL 0.0 es por el estado final, donde el getValue devuelve NA
        #Ya que las posiciones ilegales no están en el dict
        return 0.0 if q_val == -np.inf else q_val 
        
        """
        ###############################################
        ###############################################
        ###############################################
        """
        
        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        
        """
        ###############################################
        ###############################################
        ###############################################
        """
        
        if len(self.getLegalActions(state)) is 0:
            return None
        
        #Si hay accciones posibles...

        valor = self.computeValueFromQValues(state)
        acciones = []
        
        for accion in self.getLegalActions(state):
            if valor is self.getQValue(state, accion):
                acciones.append(accion)

        return random.choice(acciones)
        

        #El práctico decía usar random.choice !!        
        # #por cada accion posible
        # for accion in self.getLegalActions(state): 
        #     if valor == self.getQValue(state, accion):
        #         acciones.append([accion, valor])

        # acciones = np.array(acciones)
        # accion_opt= np.argmax(acciones, axis=0)
        
        # return acciones[accion_opt[0]][0]

        """
        ###############################################
        ###############################################
        ###############################################
        """        
        util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        """
        ###############################################
        ###############################################
        ###############################################
        """ 
        if util.flipCoin(self.epsilon): #Hint
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)        
        """
        ###############################################
        ###############################################
        ###############################################
        """                 
        return action
              
        util.raiseNotDefined()



    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        """
        ###############################################
        ###############################################
        ###############################################
        """ 
        #Probablemente este sea la línea mas importante de
        #todo el codigo
        
        #Valor viejo
        self.values[(state, action)] = (1-self.alpha) * self.values[(state,action)] 
        #Valor nuevo con el reward y descuento gamma
        self.values[(state, action)] += self.alpha * (reward + self.discount*self.computeValueFromQValues(nextState))

        """
        ###############################################
        ###############################################
        ###############################################
        """ 
        #Comento porque no tiene return        
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        """
        ###############################################
        ###############################################
        ###############################################
        """ 
        pesos = self.getWeights()
        feat_vec= self.featExtractor.getFeatures(state,action)

        return pesos*feat_vec
        
        """
        ###############################################
        ###############################################
        ###############################################
        """                 
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        """
        ###############################################
        ###############################################
        ###############################################
        """ 
        diff = (reward+self.discount*self.computeValueFromQValues(nextState)) - self.getQValue(state,action)
        feat_vec = self.featExtractor.getFeatures(state,action)

        for feat in feat_vec.keys():
            self.weights[feat] += self.alpha * diff * feat_vec[feat]
        
        """
        ###############################################
        ###############################################
        ###############################################
        """                 
        #Lo commento porque no tiene return 
        #util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            """
            ###############################################
            ###############################################
            ###############################################
            """    
            #print(self.getWeights())         
            """
            ###############################################
            ###############################################
            ###############################################
            """             
            pass
          
