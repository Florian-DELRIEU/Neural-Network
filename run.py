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
X = np.array([0.2])# données d'entrée
y = 2*X # valeurs cibles

"""
Entraînez le système neuronal en appelant la méthode train de l'instance de NeuralNetwork :
"""

learning_rate = 0.01 # taux d'apprentissage
epochs = 100 # nombre d'itérations d'entraînement
Network.train(X, y, learning_rate, epochs)

"""
Évaluez la performance du système neuronal en appelant la méthode predict de l'instance de NeuralNetwork
sur l'ensemble de validation :
"""
X_val = np.array([0.4])
y_val = np.array([0.8])

"""
Analysez les résultats de l'évaluation. Dans cet exemple, nous allons calculer le score de précision 
pour voir à quelle fréquence le système neuronal prédit correctement les valeurs cibles :
"""

#accuracy = np.mean(y_pred == y_val.argmax(axis=1))
#print("Accuracy:", accuracy)
print("Output", Network.output)

y_pred = Network.predict(X_val)