import numpy as np
import time
import matplotlib.pyplot as plt

DEBUG_ON = False
PLOTTING_ON = False

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.ones((input_size,hidden_size))  # poids de la couche d'entrée
        self.weights2 = np.ones((hidden_size, output_size))  # poids de la couche cachée
        self.bias1 = np.zeros((1, hidden_size))  # biais de la couche d'entrée (constantes)
        self.bias2 = np.zeros((1, output_size))  # biais de la couche cachée

    def function(self, x, fonction_to_use="linear"):
        """
        Calcule la fonction d'activation sigmoid sur une entrée x.
            - x (numpy.ndarray): Un scalaire, un vecteur ou une matrice d'entrée.

        Returns:
            - numpy.ndarray: La sortie de la transformation sigmoid appliquée à chaque élément de x.
        """
        if fonction_to_use == "linear": return x
        if fonction_to_use == "tanh": return np.tanh(x)
        if fonction_to_use == "sigmoid": return 1/(1 + np.exp(-x))

    def forward(self, X):
        """
        Calcule la propagation avant du réseau de neurones.

        Args:
            X (numpy.ndarray): La matrice d'entrée du réseau de neurones.

        Returns:
            numpy.ndarray: La sortie du réseau de neurones.
        """
        self.hidden = self.function(X*self.weights1 + self.bias1)  # activation de la couche cachée
        self.output = self.function(self.hidden*self.weights2 + self.bias2)  # activation de la couche de sortie

    def backward(self, X, y, learning_rate):
        """
        Calcule la rétropropagation du gradient et met à jour les poids et les biais du réseau de neurones.
        Args:
            X (numpy.ndarray): La matrice d'entrée du réseau de neurones.
            y (numpy.ndarray): Le vecteur de sortie attendu du réseau de neurones.
            learning_rate (float): Le taux d'apprentissage du réseau de neurones.
        """
        # rétropropagation du gradient
        d_output = (y - self.output)/y
        d_hidden = d_output*self.weights2.T/self.hidden

        # mise à jour des poids et des biais
        self.weights2 += learning_rate*np.dot(self.hidden.T, d_output)
        self.weights1 += learning_rate*np.dot(X.T, d_hidden)
        self.bias2 += learning_rate*np.sum(d_output, axis=0)
        self.bias1 += learning_rate*np.sum(d_hidden, axis=0)

    def train(self, X, y, learning_rate, epochs):
        """
        Entraîne le réseau de neurones en utilisant les données d'entrée `X` et les valeurs cibles `y`.

        Args:
            X (ndarray): Ensemble de données d'entraînement. Chaque ligne représente une observation, chaque colonne une caractéristique.
            y (ndarray): Valeurs cibles correspondant à chaque observation dans `X`.
            learning_rate (float): Taux d'apprentissage utilisé pour mettre à jour les poids du réseau.
            epochs (int): Nombre d'itérations d'entraînement à effectuer.
        """
        for i in range(epochs):
            if DEBUG_ON: print(f"Training... epoch {i}")
            self.forward(X)
            self.backward(X, y, learning_rate)
            if PLOTTING_ON:
                self.plot_output(i,self.output,"Output")
                self.plot_output(i,self.weights1,"Weigths")
                self.plot_output(i,self.weights2,"Weigths")


    def predict(self, X):
        self.forward(X)
        return self.output

    def plot_output(self,values,i,fig_name=None):
        if fig_name is None:    plt.figure()
        else:                   plt.figure(fig_name)
        plt.plot(values,i,"kx")
        print(self.output)