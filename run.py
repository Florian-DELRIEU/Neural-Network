from neural_network import NeuralNetwork
import numpy as np

input_size = 1 # taille de la couche d'entrée
hidden_size = 2 # taille de la couche cachée
output_size = 1 # taille de la couche de sortie
neural_network = NeuralNetwork(input_size, hidden_size, output_size)


"""
Préparez vos données d'entraînement. Dans cet exemple, nous allons utiliser des données générées aléatoirement. 
Vous pouvez utiliser vos propres données en les chargeant depuis un fichier ou en les générant de manière programmée. 
Nous allons également diviser les données en ensembles d'entraînement et de validation :
"""
X = np.array([1,2,3,4,5]).reshape(5,1) # données d'entrée
y = 2*X # valeurs cibles
X_train, X_val = X, X[::-1] # données d'entraînement et de validation
y_train, y_val = y, y[::-1] # valeurs cibles d'entraînement et de validation


"""
Entraînez le système neuronal en appelant la méthode train de l'instance de NeuralNetwork :
"""

learning_rate = 0.1 # taux d'apprentissage
epochs = 10 # nombre d'itérations d'entraînement
neural_network.train(X_train, y_train, learning_rate, epochs)

"""
Évaluez la performance du système neuronal en appelant la méthode predict de l'instance de NeuralNetwork
sur l'ensemble de validation :
"""

y_pred = neural_network.predict(X_val)

"""
Analysez les résultats de l'évaluation. Dans cet exemple, nous allons calculer le score de précision 
pour voir à quelle fréquence le système neuronal prédit correctement les valeurs cibles :
"""

accuracy = np.mean(y_pred == y_val.argmax(axis=1))
print("Accuracy:", accuracy)
print("Output", neural_network.output)