# -*- coding: utf-8 -*-
"""
Code to generate distribution in network. See the main part to set the parameters
"""

from module.neural_network_module_pytorch.build_nn import NeuralNetwork, train_model_and_save, load_model_and_optimizer, evaluate_model

import numpy as np
import sys
            
from module.quantum_network_module.quantum_network import quantum_network_fct
from module.quantum_network_module.utils.display import pretty_print

from networks import get_network
from measurements import get_measurement, TCfamily

from states import get_state
#######################################################

def save_to_file(d, path):
    f= open(path, 'w')
    np.savetxt(f, np.reshape(d, [1,np.size(d)]))
    f.close()


def type_measurement(i):
    """return the number of sources each party is connected to for a given id_network"""
    if i==0:
        return [2, 2, 2]
    elif i==1:
        return [2, 2, 2, 2]
    elif i==2:
        return [3, 2, 3, 2]
    elif i==3:
        return [3, 3, 3, 3]
    elif i==4:
        return [2, 2, 2, 2, 2]
    
def n_states(i):
    """number of bipartite states for a given id_network"""
    if i==0:
        return 3
    elif i==1:
        return 4
    elif i==2:
        return 5
    elif i==3:
        return 6
    elif i==4:#pentagon
        return 5
    
def n_party(i):
    """number of party for a given id_network"""
    out = [3,4,4,4,5]
    return out[i]

    
if __name__=="__main__":
    
    ## choose location to save the distribution
    folder="target_distributions/test/"
    
    ## create the folder 
    import os
    model_dir = os.path.dirname(folder)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    

    ## choose topology of the network
    ## 0: triangle, 1:square, 2:square with a diagonal, 3:square with two diagonal, 4:pentagon, 5:binary triangle
    config, dict_sources, dict_party = get_network(0)
    
    ## choose the state distributed by the sources (see states.py for some default states). 
    ## states should be given as density matrices.
    ## v is the visibility of the state mixed with white noise
    ## the format is a Tuple of the density matrix and a list of the dimensions of each subsystem: (density_matrix, [dimension_subsystem1, dimension_subsystem2, ...])
    sources = [get_state(0, v=1) for i in range(n_states(0))]
    
    ## choose the measurement performed by the parties (see measurements.py for some default measurements)
    measurements = [TCfamily(np.sqrt(0.785)) for i in range(n_party(0))]
    
    ## create the tuple with the name of the party and its measurements
    name_party = ["a", "b", "c", "d", "e"]
    list_measurements = [(name_party[i], measurements[i]) for i in range(n_party(0))]
    
    ## compute the target distribution
    ## if needed, specify the permutation for the systems of the sources to match the measurement
    ## if permutation=None, the function will create a permutation, it may leads to unexpected results
    permutation_povm =[0,1,2,3,4,5]
    permutation_sources=[3,4,0,5,1,2]
    permutation = [permutation_povm, permutation_sources]
    target_distribution = quantum_network_fct(dict_sources, sources, list_measurements, dict_party, permutation=permutation)

    ## save the target distribution in a file
    name_distribution = "rgb4_umax.txt"
    path = folder+name_distribution
    save_to_file(target_distribution, path)

