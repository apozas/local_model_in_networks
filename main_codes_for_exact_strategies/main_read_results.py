# -*- coding: utf-8 -*-
"""
example on how to read the model we found with main_find_local_models.py
"""

from module.neural_network_module_pytorch.build_nn import NeuralNetwork, get_dist_from_model, plot_distribution, load_model_and_optimizer, get_total_outputs, get_strategy, evaluate_model

# from distributions_pretty_elegant import pretty_elegant_distr
# from get_precomputed_distr import get_precomputed_distr

from networks import get_network

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import torch

def get_precomputed_distr(path_to_distr):
    f= open(path_to_distr, "r")
    out = np.loadtxt(f)
    f.close()
    return out

 
        
def plot_simple(d, xmin=0, xmax=np.pi/2, x=None, title=None, xlabel=None, ylabel=None, ymin=None):
    if x is None:
        x = np.linspace(xmin,xmax, len(d))
    plt.plot(x, d)
    if title is not None:
        # print(title)
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if ymin is not None:
        plt.ylim(bottom=ymin)
    plt.show()
    

def get_target_distribution(path):
    f=open(path)
    out = np.loadtxt(f)
    f.close()
    return out

def plot_all_strats(id_network, n_outputs, model, cardinality, path, width=60, depth=4):
    config,_,_ = get_network(id_network, n_outputs)
    model = NeuralNetwork(config, width, cardinality, depth=depth)
    
    from module.neural_network_module_pytorch.build_nn import plot_strats
    
    for party in config:
        source_x = config[party][0][0]
        source_y = config[party][0][1]
        plot_strats(model, path, party=party, source_x=source_x, source_y=source_y)

if __name__=="__main__":

    path = "model/test/model_test_card5/model_20.pth"
    path = "model/binary_triangle/tamas_binary_distr_card5/model_1.pth"
    
    model = NeuralNetwork()
    load_model_and_optimizer(model, path)
    distance, distribution = evaluate_model(model, dist="eucl")
    plot_distribution(model, title_style=1, dist="kl")
    # output = get_total_outputs(model)
    # strategy = get_strategy(model)
    
    from module.neural_network_module_pytorch.build_nn import plot_strats, plot_strats_binary
    # plot_strats(party="c", source_x="alpha", source_y="beta", model=model)

    plot_strats_binary(party="b", source_x="alpha", source_y="gamma", model=model)



