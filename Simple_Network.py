"""Code inspiré du réseau généré par GPT"""
import numpy as np


class NeuralNetwork:
    def __init__(self,input_size,layer1_size,layer2_size):
        # Poids des synapses
        self.weight1 = np.ones((input_size,layer1_size))
        self.weight2 = np.ones((layer1_size,layer2_size))
        # Constantes des neurones
        self.biais1 = np.zeros((1,layer1_size))
        self.biais2 = np.zeros((1,layer2_size))

    def function(self,x,used_function="tanh"):
        """
        Determine the function to use as activation function
        :param x:
        :param used_function:
                - tanh: tangeant hyperbolique
        """
        if used_function == "tanh": return np.tanh(x)
        else: return NameError , "Wrong function name"

