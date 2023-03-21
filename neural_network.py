import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)  # poids de la couche d'entrée
        self.weights2 = np.random.randn(hidden_size, output_size)  # poids de la couche cachée
        self.bias1 = np.zeros((1, hidden_size))  # biais de la couche d'entrée
        self.bias2 = np.zeros((1, output_size))  # biais de la couche cachée

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def forward(self, X):
        self.hidden = self.sigmoid(np.dot(X, self.weights1) + self.bias1)  # activation de la couche cachée
        self.output = self.sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)  # activation de la couche de sortie

    def backward(self, X, y, learning_rate):
        # rétropropagation du gradient
        d_output = (y - self.output)*self.output*(1 - self.output)
        d_hidden = np.dot(d_output, self.weights2.T)*self.hidden*(1 - self.hidden)

        # mise à jour des poids et des biais
        self.weights2 += learning_rate*np.dot(self.hidden.T, d_output)
        self.weights1 += learning_rate*np.dot(X.T, d_hidden)
        self.bias2 += learning_rate*np.sum(d_output, axis=0)
        self.bias1 += learning_rate*np.sum(d_hidden, axis=0)

    def train(self, X, y, learning_rate, epochs):
        for i in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)

    def predict(self, X):
        self.forward(X)
        return self.output.argmax(axis=1)
