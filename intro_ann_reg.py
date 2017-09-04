# -*- coding: utf-8 -*-

# Biblioteca utilizada
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Modelo
class ArtificialNeuralNet():
    def __init__(self, input_dim, output_dim, hidden_size):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size

        np.random.seed(1)
        # Inicialização de pesos e vieses
        self.Wxh = np.random.normal(0.0, self.hidden_size ** -0.05,
                                    (self.input_dim[1], self.hidden_size))
        self.Why = np.random.normal(0.0, self.hidden_size ** -0.05,
                                    (self.hidden_size, self.output_dim[1]))

    def sigmoid(self, x, derivative=False):
        if derivative:
            return x * (1.0 - x)
        return 1.0 / (1.0 + np.exp(-x))

    def predict(self, inputs):
        self.hidden_layer = self.sigmoid(np.dot(inputs, self.Wxh))
        self.output = np.dot(self.hidden_layer, self.Why)
        return self.output

    def fit(self, inputs, targets, batch_size, epochs, display_step=False,
            learning_rate=0.01):
        self.inputs = inputs
        self.target = targets
        self.learning_rate = learning_rate

        for e in range(epochs):
            batch = np.random.choice(range(X.shape[0]), size=batch_size)
            # Forward pass
            self.predict(self.inputs[batch])

            # Backward pass
            erro = self.output - self.target[batch]
            hidden_erro = np.dot(erro, self.Why.T)
            dWxh = hidden_erro * self.sigmoid(self.hidden_layer,
                                              derivative=True)

            self.Why -= self.learning_rate * np.dot(self.hidden_layer.T, erro)
            self.Wxh -= self.learning_rate * np.dot(self.inputs[batch].T, dWxh)

            if display_step and e % display_step == 0:
                loss = self.score(erro)
                print('Erro na iteração {:d}: {:.5f}'.format(e, loss))

        return self

    def score(self, erro):
        return np.mean(erro ** 2)


if __name__ == '__main__':

    # Dados
    arquivo = os.path.join(os.getcwd(), 'hour.csv')
    dados = pd.read_csv(arquivo)

    # Preprocessamento dos dados categoricos
    dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
    for each in dummy_fields:
        dummies = pd.get_dummies(dados[each], prefix=each, drop_first=False)
        dados = pd.concat([dados, dummies], axis=1)

    fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 'weekday',
                      'atemp', 'mnth', 'workingday', 'hr']
    dados.drop(fields_to_drop, inplace=True, axis=1)

    # Preprocessamento dos dados numéricos
    quant_features = ['casual', 'registered', 'cnt',
                      'temp', 'hum', 'windspeed']
    scaled_features = {}
    for each in quant_features:
        if each in ['casual', 'registered', 'cnt']:
            dados[each] = np.log1p(dados[each])
        mean, std = dados[each].mean(), dados[each].std()
        scaled_features[each] = [mean, std]
        dados[each] = (dados[each] - mean) / std

    target_fields = ['cnt', 'casual', 'registered']
    features = dados.drop(target_fields, axis=1)
    targets = dados[target_fields]

    # Hiperparametros
    hidden_size = 10
    learning_rate = 0.01
    batch_size = 512
    epochs = 100000
    display = 10000

    # Aplicação do modelo
    X = features.values
    y = targets['cnt'].values.reshape((-1, 1))

    modelo = ArtificialNeuralNet(X.shape, y.shape, hidden_size)
    modelo = modelo.fit(X, y, batch_size, epochs, display, learning_rate)
    resultado = modelo.predict(X)

    mean, std = scaled_features['cnt']
    original = np.exp((dados['cnt'] * std) + mean) - 1
    resultado = np.exp((resultado * std) + mean) - 1
    diff = original.values.reshape(-1) - resultado.reshape(-1)
    fig, ax = plt.subplots(1, 1)
    ax.set(title='Resultado')
    plt.plot(diff)
    plt.show()
