"""
Premier essai de réseaux de neurones
"""
import numpy as np

class InitialLayer:
    def __init__(self,N):
        self.input = None
        self.output = None

    def FeedForward(self,Input):
        """
        :Input: liste de taille N (autant d'entrée que de neurones)
        """
        self.input = Input
        self.output = self.input

class Layer:
    def __init__(self,N):
        self.input = None
        self.output = None