from neural_network import NeuralNetwork
import numpy as np

input_size = 1 # taille de la couche d'entrée
hidden_size = 1 # taille de la couche cachée
output_size = 1 # taille de la couche de sortie
Network = NeuralNetwork(input_size, hidden_size, output_size)


"""
Préparez vos données d'entraînement. Dans cet exemple, nous allons utiliser des données générées aléatoirement. 
Vous pouvez utiliser vos propres données en les chargeant depuis un fichier ou en les générant de manière programmée. 
Nous allons également diviser les données en ensembles d'entraînement et de validation :
"""
Data = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14]).reshape(15,1)
X = Data[2]
Y = 2*X

"""
Entraînez le système neuronal en appelant la méthode train de l'instance de NeuralNetwork :
"""
learning_rate = 0.001 # taux d'apprentissage
epochs = 1000 # nombre d'itérations d'entraînement
Network.train(X, Y, learning_rate, epochs)


print(f"Valeurs cibles: \n {Y}")
print(f"Valeurs obtenues: \n {Network.output}")
