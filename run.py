from neural_network import NeuralNetwork
import numpy as np

input_size = 1 # taille de la couche d'entrée
hidden_size = 2 # taille de la couche cachée
output_size = 1 # taille de la couche de sortie
Network = NeuralNetwork(input_size, hidden_size, output_size)


"""
Préparez vos données d'entraînement. Dans cet exemple, nous allons utiliser des données générées aléatoirement. 
Vous pouvez utiliser vos propres données en les chargeant depuis un fichier ou en les générant de manière programmée. 
Nous allons également diviser les données en ensembles d'entraînement et de validation :
"""
X = np.array([0.2])# données d'entrée
y = 2*X # valeurs cibles

"""
Entraînez le système neuronal en appelant la méthode train de l'instance de NeuralNetwork :
"""
learning_rate = 0.1 # taux d'apprentissage
epochs = 200 # nombre d'itérations d'entraînement
Network.train(X, y, learning_rate, epochs)

