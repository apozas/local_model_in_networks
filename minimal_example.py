# -*- coding: utf-8 -*-
"""
Minimal example to run the neural network to find local models via sampling the local distribution
"""

from module.nn_sampling.build_nn import NeuralNetwork, train_model_and_save

#configuration of the network
config = {
    'a': (['beta', 'gamma'], 2),
    'b': (['alpha', 'gamma'], 2),
    'c': (['alpha', 'beta'], 2)
}

#choose a target distribution
target_distribution = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3]

#build the neural network
model = NeuralNetwork(config, target_distribution)

#train the model
train_model_and_save(model)

