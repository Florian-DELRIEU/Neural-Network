"""Code inspiré du réseau généré par GPT"""
import numpy as np


class NeuralNetwork:
    def __init__(self,input_size,layer1_size,layer2_size):
        self.weight1 = np.ones((input_size,layer1_size))
        self.weight2 = np.ones((layer1_size,layer2_size))