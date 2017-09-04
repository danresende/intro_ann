# -*- coding: utf-8 -*-

# Biblioteca utilizada
import numpy as np


# Modelo
class ArtificialNeuralNet():
    def __init__(self, input_dim, output_dim, hidden_size):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size

        np.random.seed(1)
        # Inicialização de pesos e vieses
        self.Wxh = 2 * np.random.random((self.input_dim[1],
                                         self.hidden_size)) - 1
        # self.Whh = np.random.randn(self.hidden_size, self.hidden_size)
        self.Why = 2 * np.random.random((self.hidden_size,
                                         self.output_dim[1])) - 1

    def sigmoid(self, x, derivative=False):
        if derivative:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        self.hidden_layer = self.sigmoid(np.dot(inputs, self.Wxh))
        self.output = self.sigmoid(np.dot(self.hidden_layer, self.Why))
        return self.output

    def fit(self, inputs, targets, epochs,
            display_step=False, learning_rate=0.01):
        self.inputs = inputs
        self.target = targets
        self.learning_rate = learning_rate

        for e in range(epochs):
            # Forward pass
            self.predict(self.inputs)

            # Backward pass
            error = self.output - self.target
            dWhy = error * self.sigmoid(self.output, derivative=True)
            dWxh = np.dot(dWhy, self.Why.T) * self.sigmoid(self.hidden_layer,
                                                           derivative=True)
            self.Why -= self.learning_rate * np.dot(self.hidden_layer.T, dWhy)
            self.Wxh -= self.learning_rate * np.dot(self.inputs.T, dWxh)

            if display_step and e % display_step == 0:
                loss = self.score(error)
                print('Erro na iteração {:d}: {:.5f}'.format(e, loss))

        return self

    def score(self, erro):
        return np.mean(np.abs(erro))


if __name__ == '__main__':

    # Dados
    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    y = np.array([[0, 1, 1, 0]]).T

    # Hiperparametros
    hidden_size = 32
    learning_rate = 0.01

    # Aplicação do modelo
    modelo = ArtificialNeuralNet(X.shape, y.shape, hidden_size)
    modelo = modelo.fit(X, y, 60000, 10000, learning_rate)
    resultado = modelo.predict(X)
    print(resultado)
