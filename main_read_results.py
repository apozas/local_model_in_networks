# -*- coding: utf-8 -*-
"""
example on how to read the model we found with main_find_local_models.py
"""

from module.neural_network_module_pytorch.build_nn import NeuralNetwork, get_dist_from_model, evaluate_model_return_distribution, evaluate_model_return_dist, plot_distribution, load_model_and_optimizer

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

def read_simple(save_path, id_network=0, no_checkpoint=False, target_distribution=None, target_distribution_path=None, n_sample = 50000, width=60, depth=4, dist="kl"):
    if target_distribution_path is not None:
        target_distribution=get_precomputed_distr(target_distribution_path)
        
    config,_,_ = get_network(id_network)
    model = NeuralNetwork(config, width, num_hidden_layers=depth)
    if no_checkpoint:
        return evaluate_model_return_dist(model, save_path, n_sample, target_distribution, 3, dist=dist)
    else:
        return get_dist_from_model(model, save_path)

        
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
    
    
def get_distr_from_model(path, id_network, width, depth, n_samples):
    """return the distribution of a local model"""
    config,_,_ = get_network(id_network)
    model = NeuralNetwork(config, width, num_hidden_layers=depth)
    target_distribution = get_precomputed_distr(path)
    distr=evaluate_model_return_distribution(model, n_samples, target_distribution, opt=3, path=path)
    return distr


def plot_distr_simple(id_network, n_samples, width, depth, path, target_path, dist):
    """for distribution from a given path"""
    config, dict_sources, dict_party = get_network(id_network)
    
    model = NeuralNetwork(config, width, num_hidden_layers=depth)

    target_distribution=get_precomputed_distr(target_path)
    plot_distribution(model, path, n_samples, target_distribution, opt=3, dist=dist)
    
    distr = evaluate_model_return_distribution(model, n_samples, target_distribution, opt=3, path=path)
    return distr

def get_target_distribution(path):
    f=open(path)
    out = np.loadtxt(f)
    f.close()
    return out



if __name__=="__main__":
    
    ## get the distance of a saved model (with no_checkpoints=False, no target distributions are needed, dist and n_sample are ignored)
    d=read_simple("model/test/model_test.pth", id_network=0, target_distribution_path="target_distributions/test/rgb4_umax.txt", no_checkpoint=True, dist="eucl", n_sample = 1000000)
    
    ## plot the local and target distribution
    distribution=plot_distr_simple(id_network=0, n_samples=1000000, width=60, depth=4, path="model/test/model_test.pth", target_path="target_distributions/test/rgb4_umax.txt", dist="eucl")
    
