# -*- coding: utf-8 -*-
"""
example on how to read the model we found with main_find_local_models.py
"""

from module.nn_sampling.build_nn import NeuralNetwork, get_dist_from_model, evaluate_model_return_distribution, evaluate_model_return_dist, plot_distribution, load_model_and_optimizer
from module.nn_sampling.build_nn import evaluate_model

import numpy as np
import matplotlib.pyplot as plt 


def get_target_distribution(path):
    """read a distribution from a file, returns it as an array"""
    f=open(path)
    out = np.loadtxt(f)
    f.close()
    return out


if __name__=="__main__":
    
    save_path="model/test/model_test.pth"
    
    ## main function to evaluate a model:
        ## from a path
    dist, distribution = evaluate_model(n_samples=10000, path=save_path)
    
        ## from a loaded model
    model = NeuralNetwork()
    load_model_and_optimizer(model, save_path)
    dist, distribution = evaluate_model(n_samples=10000, model=model)

    ## get the distance of a saved model (by evaluating it)
    ## the type of distance (dist) can be "eucl" : Euclidean distance or "kl": Kullback–Leibler divergence 
    dist = evaluate_model_return_dist(n_samples=10000, path=save_path, dist="kl")
    
    ## get the distance of a saved model (by reading the save best distance during training as a checkpoint)
    dist = get_dist_from_model(save_path)
    
    ## or equivalently
    model = NeuralNetwork()
    load_model_and_optimizer(model, save_path)
    dist = model.best_dist

    ## get the distribution built by the model
    distribution = evaluate_model_return_distribution(n_samples=10000, path=save_path)

    ## plot the local and target distribution
        ## from a path
    plot_distribution(n_samples=100000, path=save_path, dist="kl", title_style=1)
    plt.show()
    
        ## from a loaded model
    model = NeuralNetwork()
    load_model_and_optimizer(model, save_path)
    plot_distribution(n_samples=100000, model=model, dist="kl", title_style=1)
    plt.show()
    
    ## display local strategies
    model = NeuralNetwork()

    load_model_and_optimizer(model, "model/test/model_test.pth")
    # load_model_and_optimizer(model, "model/tetrahedron/model.pth")
    
    from module.nn_sampling.build_nn import plot_strats
    ## the party, source_x and source_y should be the name as strings as defined in model.config
    
    ## example for the triangle
    data, prediction = plot_strats(model, party="a", source_x="beta", source_y="gamma", n_points=1000, 
                                    set_sources_value={}, uniform_local_variables=True, plot_most_likely_outcomes=False)
    
    ## example for the tetrahedron
    # data, prediction = plot_strats(model, party="a", source_x="ac", source_y="ad", n_points=1000, 
    #                                 set_sources_value={"ab":0.2}, uniform_local_variables=True, plot_most_likely_outcomes=False)
