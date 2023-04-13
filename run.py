from neural_network import NeuralNetwork
import numpy as np

input_size = 1 # taille de la couche d'entrée
hidden_size = 3 # taille de la couche cachée
output_size = 1 # taille de la couche de sortie
Network = NeuralNetwork(input_size, hidden_size, output_size)


"""
Préparez vos données d'entraînement. Dans cet exemple, nous allons utiliser des données générées aléatoirement. 
Vous pouvez utiliser vos propres données en les chargeant depuis un fichier ou en les générant de manière programmée. 
Nous allons également diviser les données en ensembles d'entraînement et de validation :
"""
Data = np.array([ 0.45375244, -0.87447848,  0.20418979,  0.22612347, -0.82194152,
        0.76487173, -0.01377334, -0.25665163, -0.72822261, -0.44119749,
        0.87519806, -0.94695929, -0.53262514, -0.7910783 , -0.01285381,
        0.32680019,  0.04060011, -0.58239623, -0.52266407, -0.83104156,
       -0.06727012, -0.80390909,  0.24674281,  0.26055366, -0.1371677 ,
       -0.88861433,  0.19587372,  0.98332952, -0.21530903,  0.64215399]).reshape(30,1)
X = Data[:4]
y = np.where(X < -0.3, -1, np.where(X > 0.3, 1, 0))

"""
Entraînez le système neuronal en appelant la méthode train de l'instance de NeuralNetwork :
"""
learning_rate = 0.2 # taux d'apprentissage
epochs = 1000 # nombre d'itérations d'entraînement
Network.train(X, y, learning_rate, epochs)


print(f"Valeurs cibles: \n {y}")
print(f"Valeurs obtenues: \n {Network.output}")
