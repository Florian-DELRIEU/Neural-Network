import numpy as np
import matplotlib.pyplot as plt
import MyPack2.Utilities as util

# Parametres Utilisateur
DEBUG = True        # Affiche des logs ?
PLOTTING = True    # Trace dans graphique
SAVING = True       # Sauvegarde data

# Parametres machine
IS_SAVING_DATA = PLOTTING or SAVING

ACTIVATION_FUNCTION = "tanh"

class NeuralNetwork:
    """
    TODO
        - Pouvoir sauvegarder une partie des données (1/100 par ex.)
            - il faut une nouvelle fonction avec en argument les variables a sauvegarder
    """
    def __init__(self, input_size, hidden_size, output_size):
        # Poids initiaux
        self.weights1 = np.ones((input_size,hidden_size))  # poids de la couche d'entrée
        self.weights2 = np.ones((hidden_size,output_size))  # poids de la couche caché
        # Constantes
        self.bias1 = np.zeros((1, hidden_size))  # biais de la couche d'entrée (constantes)
        self.bias2 = np.zeros((1, output_size))  # biais de la couche cachée
        # Sorties des couches
        self.hidden = np.array([])
        self.output = np.array([])

        self.list_weights1 = []
        self.list_weights2 = []
        self.list_bias1 = []
        self.list_bias2 = []
        self.list_hidden = []
        self.list_output = []

    def function(self, x, fonction_to_use= ACTIVATION_FUNCTION):
        """
        Calcule la fonction d'activation sigmoid sur une entrée x.
            - x (numpy.ndarray): Un scalaire, un vecteur ou une matrice d'entrée.

        Returns:
            - numpy.ndarray: La sortie de la transformation sigmoid appliquée à chaque élément de x.
        """
        if fonction_to_use == "linear":     return x
        if fonction_to_use == "tanh":       return np.tanh(x)
        if fonction_to_use == "sigmoid":    return 1/(1 + np.exp(-x))

    def forward(self, X):
        """
        Calcule la propagation avant du réseau de neurones.

        Args:
            X (numpy.ndarray): La matrice d'entrée du réseau de neurones.

        Returns:
            numpy.ndarray: La sortie du réseau de neurones.
        """
        self.hidden = 10*self.function(np.dot(X,self.weights1) + self.bias1)  # activation de la couche cachée
        self.output = 10*self.function(np.dot(self.hidden,self.weights2) + self.bias2)  # activation de la couche de sortie

        # Sauvegarde des données ?
        if IS_SAVING_DATA:
            self.list_hidden.append(self.hidden.copy())
            self.list_output.append(self.output.copy())

    def backward(self, X, y, learning_rate):
        """
        Calcule la rétropropagation du gradient et met à jour les poids et les biais du réseau de neurones.
        Args:
            X (numpy.ndarray): La matrice d'entrée du réseau de neurones.
            y (numpy.ndarray): Le vecteur de sortie attendu du réseau de neurones.
            learning_rate (float): Le taux d'apprentissage du réseau de neurones.
        """
        # rétropropagation du gradient
        if ACTIVATION_FUNCTION == "sigmoid":
            d_output = (y - self.output)*self.output*(1 - self.output)
            d_hidden = np.dot(d_output, self.weights2.T)*self.hidden*(1 - self.hidden)
        elif ACTIVATION_FUNCTION == "tanh":
            d_output = (y - self.output) * (1 - self.output ** 2)
            d_hidden = np.dot(d_output, self.weights2.T) * (1 - self.hidden**2)


        # mise à jour des poids et des biais
        self.weights2 += learning_rate*np.dot(self.hidden.T,d_output)
        self.weights1 += learning_rate*np.dot(X.T,d_hidden)
        self.bias2 += learning_rate*np.sum(d_output, axis=0)
        self.bias1 += learning_rate*np.sum(d_hidden, axis=0)

        # Sauvegarde des données ?
        if IS_SAVING_DATA:
            self.list_weights1.append(self.weights1.copy())
            self.list_weights2.append(self.weights2.copy())
            self.list_bias1.append(self.bias1.copy())
            self.list_bias2.append(self.bias2.copy())

    def train(self, X, y, learning_rate, epochs):
        """
        Entraîne le réseau de neurones en utilisant les données d'entrée `X` et les valeurs cibles `y`.

        Args:
            X (ndarray): Ensemble de données d'entraînement. Chaque ligne représente une observation, chaque colonne une caractéristique.
            y (ndarray): Valeurs cibles correspondant à chaque observation dans `X`.
            learning_rate (float): Taux d'apprentissage utilisé pour mettre à jour les poids du réseau.
            epochs (int): Nombre d'itérations d'entraînement à effectuer.
        """
        if DEBUG: print("Training ...")
        for i in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            if DEBUG: util.progress_print(i, epochs, int(epochs/10))
        if PLOTTING:
            self.plot_output(self.list_weights1,"Weigths1")
            self.plot_output(self.list_bias1,"Biais1")
            self.plot_output(self.list_hidden,"Hidden")
            self.plot_output(self.list_weights2,"Weigths2")
            self.plot_output(self.list_bias2,"Biais2")
            self.plot_output(self.list_output, "Output")
#            if DEBUG_ON: print(self.output)
        print("Trainig done")


    def predict(self, X):
        self.forward(X)
        return self.output

    def plot_output(self,values,fig_name=None):
        values = np.array(values)
        if fig_name is None:    plt.figure()
        else:                   plt.figure(fig_name)
        # Recupere le shape de values
        reordered_values = util.reorder_array(values)
        if len(reordered_values.shape) == 2:
            for i in range(reordered_values.shape[0]):  # Pour chaque éléments
                epochs = reordered_values.shape[-1]
                plt.plot(reordered_values[i].reshape(epochs),"-")
        if len(reordered_values.shape) == 3:
            for i in range(reordered_values.shape[0]):
                for j in range(reordered_values.shape[1]):
                    epochs = reordered_values.shape[-1]
                    plt.plot(reordered_values[i,j].reshape(epochs),"-")

    def input(self,X):
        assert type(X) is int
        np.array([X])
        self.predict(X)

    def set_weight(self, shape, a, b):
        """
        This function returns a numpy array of the given shape with equidistant values between a and b.
        """
        size = np.prod(shape)
        delta = (b - a) / (size - 1)
        values = np.arange(size) * delta + a
        return np.reshape(values, shape)

