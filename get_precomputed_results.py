# -*- coding: utf-8 -*-
"""
file to read some results presented in the report 

see the __main__ at the end of the file to read the results
"""

from module.neural_network_module_pytorch.build_nn import NeuralNetwork, train_model_and_save, load_model_and_optimizer, evaluate_model
from module.neural_network_module_pytorch.build_nn import get_dist_from_model, evaluate_model_return_distribution, evaluate_model_return_dist, plot_distribution

import numpy as np
import sys
            
from module.quantum_network_module.quantum_network import quantum_network_fct
from module.quantum_network_module.utils.display import pretty_print

from networks import get_network
from measurements import get_measurement
from states import get_state


from networks import get_network

import os
import matplotlib.pyplot as plt
import matplotlib
#######################################################

def save_to_file(d, folder, id_network, id_measurement, id_state, id_visibility):
    f= open("{}/distributions_{}_{}_{}_{}.txt".format(folder, id_network, id_measurement, id_state, id_visibility), 'w')
    np.savetxt(f, np.reshape(d, [1,np.size(d)]))
    f.close()

def save_to_file_elegant_family(d, id_network, id_measurement, id_state, id_visibility):
    f= open("all_elegant_d/distributions_{}_{}_{}_{}.txt".format(id_network, id_measurement, id_state, id_visibility), 'w')
    np.savetxt(f, np.reshape(d, [1,np.size(d)]))
    f.close()
    
# def get_precomputed_distr(id_network, id_measurement, id_state, id_visibility):
#     f= open("all_distributions/distributions_{}_{}_{}_{}.txt".format(id_network, id_measurement, id_state, id_visibility), 'r')
#     out = np.loadtxt(f)
#     f.close()
#     return out

def get_precomputed_distr(path_to_distr):
    f= open(path_to_distr, "r")
    out = np.loadtxt(f)
    f.close()
    return out

def run_NN(model, target_distribution, id_network, id_measurement, id_state, id_visibility):
    save_path = 'model/bigsearch/model_{}_{}_{}_{}_opt3.pth'.format(id_network, id_measurement, id_state, id_visibility)

    # first round ---
    n_value_total = 5000
    batch_size = n_value_total
    
    # load_model_and_optimizer(model, save_path)
    
    train_model_and_save(model, n_value_total, target_distribution, batch_size, 2000, model_path=save_path, value_disp=100000, opt=3, critere="kl")

    # second round ---
    n_value_total = 100000
    batch_size = n_value_total
    
    load_model_and_optimizer(model, save_path)
    
    train_model_and_save(model, n_value_total, target_distribution, batch_size, 100, model_path=save_path, value_disp=100000, opt=3, critere="kl")

    # third round ---
    n_value_total = 300000
    batch_size = n_value_total
    
    load_model_and_optimizer(model, save_path)
    
    train_model_and_save(model, n_value_total, target_distribution, batch_size, 5000, model_path=save_path, value_disp=100000, opt=3, critere="eucl")


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

def gen_elegant_BSM():
    from measurements import elegant_family
    
    theta = np.linspace(0,np.pi/2, 21)
    
    for id_network in [0,]:
        for t in range(21):
            for id_state in [3,]:
                for id_visibility in [0,1,2]:
                    
                    all_visibility= [1, 0.99, 0.95]
                    visibility = all_visibility[id_visibility]
                    config, dict_sources, dict_party = get_network(id_network)
                    
                    sources = [get_state(id_state, visibility) for i in range(n_states(id_network))]
                    measurements = [elegant_family(theta[t]) for i in range(n_party(id_network))]
                    
                    name_party = ["a", "b", "c", "d"]
                    
                    list_measurements = [(name_party[i], measurements[i]) for i in range(n_party(id_network))]
                    
                    target_distribution = quantum_network_fct(dict_sources, sources, list_measurements, dict_party)
                
                    save_to_file_elegant_family(target_distribution, id_network, t, id_state, id_visibility)
                
        
    def gen_bs(folder, id_network, id_measurement, id_state, id_visibility, permutation=None):
        import os
        model_dir = os.path.dirname(folder)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        all_visibility= [1, 0.99, 0.95]
        visibility = all_visibility[id_visibility]
        config, dict_sources, dict_party = get_network(id_network)
        
        sources = [get_state(id_state, visibility) for i in range(n_states(id_network))]
        measurements = [get_measurement(id_measurement) for i in range(n_party(id_network))]
        
        name_party = ["a", "b", "c", "d"]
        
        list_measurements = [(name_party[i], measurements[i]) for i in range(n_party(id_network))]
        
        target_distribution = quantum_network_fct(dict_sources, sources, list_measurements, dict_party, permutation=permutation)
    
        # model = NeuralNetwork(config, 32, num_hidden_layers=3)
        save_to_file(target_distribution, folder, id_network, id_measurement, id_state, id_visibility)
        
def make_ws_distr():
    folder = "test/"
    
    import os
    model_dir = os.path.dirname(folder)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    permutation_povm =[0,1,2,3,4,5]
    permutation_sources=[3,4,0,5,1,2] #or invert 2 and 3 for measurement of Bob on gamma first
    permutation_triangle = [permutation_povm, permutation_sources]
    permutation_triangle = None

    permutation_povm =[0,1,2,3,4,5,6,7]
    permutation_sources=[7,0,1,2,3,4,5,6]
    permutation_square = [permutation_povm, permutation_sources]
    all_permutations = [permutation_triangle, permutation_square]
    for id_network in [0,]:
        for id_measurement in [0,1,2,3,4,5]:
            for id_state in [0,1,2,3,4,5]:
                for id_visibility in [0,1,2]:
                    
                    
                    all_visibility= [1, 0.99, 0.95]
                    visibility = all_visibility[id_visibility]
                    config, dict_sources, dict_party = get_network(id_network)
                    
                    sources = [get_state(id_state, visibility) for i in range(n_states(id_network))]
                    measurements = [get_measurement(id_measurement) for i in range(n_party(id_network))]
                    
                    name_party = ["a", "b", "c", "d"]
                    
                    list_measurements = [(name_party[i], measurements[i]) for i in range(n_party(id_network))]
                    
                    target_distribution = quantum_network_fct(dict_sources, sources, list_measurements, dict_party, permutation=all_permutations[id_network])
                
                    # model = NeuralNetwork(config, 32, num_hidden_layers=3)
                    save_to_file(target_distribution, folder, id_network, id_measurement, id_state, id_visibility)
   
def gen_elegant_BSM_square():
    from measurements import elegant_family
    folder="eBSM1/"
    import os
    model_dir = os.path.dirname(folder)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    permutation_povm =[0,1,2,3,4,5,6,7]
    permutation_sources=[7,0,1,2,3,4,5,6]
    permutation_square = [permutation_povm, permutation_sources]
    theta = np.linspace(0,np.pi/2, 21)
    
    for id_network in [1,]:
        for t in range(21):
            for id_state in [0,]:
                for id_visibility in [0,1,2]:
                    
                    all_visibility= [1, 0.99, 0.95]
                    visibility = all_visibility[id_visibility]
                    config, dict_sources, dict_party = get_network(id_network)
                    
                    sources = [get_state(id_state, visibility) for i in range(n_states(id_network))]
                    measurements = [elegant_family(theta[t]) for i in range(n_party(id_network))]
                    
                    name_party = ["a", "b", "c", "d"]
                    
                    list_measurements = [(name_party[i], measurements[i]) for i in range(n_party(id_network))]
                    
                    target_distribution = quantum_network_fct(dict_sources, sources, list_measurements, dict_party, permutation=permutation_square)
                
                    # model = NeuralNetwork(config, 32, num_hidden_layers=3)
                    save_to_file(target_distribution, folder, id_network, t, id_state, id_visibility)
                      
def gen_elegant_BSM_tri():
    from measurements import elegant_family
    folder="eBSM1/"
    import os
    model_dir = os.path.dirname(folder)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    theta = np.linspace(0,np.pi/2, 21)
    
    for id_network in [0,]:
        for t in range(21):
            for id_state in [0,]:
                for id_visibility in [0,1,2]:
                    
                    all_visibility= [1, 0.99, 0.95]
                    visibility = all_visibility[id_visibility]
                    config, dict_sources, dict_party = get_network(id_network)
                    
                    sources = [get_state(id_state, visibility) for i in range(n_states(id_network))]
                    measurements = [elegant_family(theta[t]) for i in range(n_party(id_network))]
                    
                    name_party = ["a", "b", "c", "d"]
                    
                    list_measurements = [(name_party[i], measurements[i]) for i in range(n_party(id_network))]
                    
                    target_distribution = quantum_network_fct(dict_sources, sources, list_measurements, dict_party)
                
                    # model = NeuralNetwork(config, 32, num_hidden_layers=3)
                    save_to_file(target_distribution, folder, id_network, t, id_state, id_visibility)
                      
                 
def gen_elegant_BSM_pentagon():
    def get_net_penta():
        config = {
            'a': (['epsilon', 'alpha'], 4),
            'b': (['alpha', 'beta'], 4),
            'c': (['beta', 'gamma'], 4),
            'd': (['gamma', 'delta'], 4),
            'e': (['delta', 'epsilon'], 4)
        }
        dict_sources = {
            'alpha': ['a', 'b'],
            'beta': ['b', 'c'],
            'gamma': ['c', 'd'],
            'delta': ['d', 'e'],
            'epsilon': ['e', 'a']
        }
        dict_party = {
            'a': ['epsilon', 'alpha'],
            'b': ['alpha', 'beta'],
            'c': ['beta', 'gamma'],
            'd': ['gamma', 'delta'],
            'e': ['delta', 'epsilon']
        }
        return config, dict_sources, dict_party
    
    from measurements import elegant_family
    folder="eBSM_penta/"
    model_dir = os.path.dirname(folder)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    permutation_povm =[0,1,2,3,4,5,6,7,8,9]
    permutation_sources=[9,0,1,2,3,4,5,6,7,8]
    permutation_square = [permutation_povm, permutation_sources]
    theta = np.linspace(0,np.pi/2, 21)
    
    for id_network in [4,]:
        for t in range(10,21):
            for id_state in [0,]:
                for id_visibility in [0,1,2]:
                    #to remove:
                    if t==10 and id_visibility==0:
                        pass
                    else:
                        all_visibility= [1, 0.99, 0.95]
                        visibility = all_visibility[id_visibility]
                        config, dict_sources, dict_party = get_network(id_network)
                        
                        sources = [get_state(id_state, visibility) for i in range(n_states(id_network))]
                        measurements = [elegant_family(theta[t]) for i in range(n_party(id_network))]
                        
                        name_party = ["a", "b", "c", "d", "e"]
                        
                        list_measurements = [(name_party[i], measurements[i]) for i in range(n_party(id_network))]
                        
                        target_distribution = quantum_network_fct(dict_sources, sources, list_measurements, dict_party, permutation=permutation_square)
                    
                        # model = NeuralNetwork(config, 32, num_hidden_layers=3)
                        save_to_file(target_distribution, folder, id_network, t, id_state, id_visibility)
                    
                    



def read_simple(save_path, id_network=0, no_checkpoint=False, target_distribution=None, target_distribution_path=None, n_sample = 50000, width=60, depth=4, dist="kl"):
    if target_distribution_path is not None:
        target_distribution=get_precomputed_distr(target_distribution_path)
        
    config,_,_ = get_network(id_network)
    model = NeuralNetwork(config, width, num_hidden_layers=depth)
    if no_checkpoint:
        return evaluate_model_return_dist(model, save_path, n_sample, target_distribution, 3, dist=dist)
    else:
        return get_dist_from_model(model, save_path)


def read_model(folder, model, id_network, id_measurement, id_state, id_visibility, no_checkpoint=False, target_distribution=None, n_sample = 50000):
        save_path = 'model/{}/model_{}_{}_{}_{}_opt3.pth'.format(folder, id_network, id_measurement, id_state, id_visibility)
        if no_checkpoint:
            path="model/{}/model_{}_{}_{}_{}_opt3.pth".format(folder, id_network, id_measurement, id_state, id_visibility)
            return evaluate_model_return_dist(model, path, n_sample, target_distribution, 3)
        else:
            return get_dist_from_model(model, save_path)
  
   
def print_analysis(d, networks=[0,], measurements=[0,], states = [0,]):
    for network in networks:
        for measurement in measurements:
            for state in states:
                print("network ", network, "measurement ", measurement, "state ", state, "\t ", d[network][measurement][state])
    

   
def read_all(folder, id_state, id_network=0, width=64, depth=3, no_checkpoint=False, n_sample=50000, dist = "kl"):
    d = np.zeros([21,3])
    
    for id_measurement in range(21):
        # for id_state in [6,]:
        for id_visibility in [0,1,2]:
            config,_,_ = get_network(id_network)
            model = NeuralNetwork(config, width, num_hidden_layers=depth, dist=dist)
            try:
                if no_checkpoint:
                    target_distribution = get_precomputed_distr("all_elegant_d/distributions_{}_{}_{}_{}.txt".format(id_network, id_measurement, id_state, id_visibility))
                    d[id_measurement][id_visibility]=read_model(folder, model, id_network, id_measurement, id_state, id_visibility, no_checkpoint, target_distribution, n_sample)
                else:
                    d[id_measurement][id_visibility]=read_model(folder, model, id_network, id_measurement, id_state, id_visibility)
            except:
                print("no model_{}_{}_{}_{}".format(id_network, id_measurement, id_state, id_visibility))
                d[id_measurement][id_visibility]=0
    return d
 
def read_all_cs(folder, id_measurement, id_network=0, width=64, depth=3, no_checkpoint=False, n_sample=50000, dist = "kl"):
    d = np.zeros([21,3])
    
    for id_state in range(21):
        # for id_state in [6,]:
        for id_visibility in [0,1,2]:
            config,_,_ = get_network(id_network)
            model = NeuralNetwork(config, width, num_hidden_layers=depth, dist=dist)
            try:
                if no_checkpoint:
                    target_distribution = get_precomputed_distr("changing_states_{}_{}/distributions_{}_{}.txt".format(id_network, id_measurement, id_state, id_visibility))
                    d[id_state][id_visibility]=read_model(folder, model, id_network, id_measurement, id_state, id_visibility, no_checkpoint, target_distribution, n_sample)
                else:
                    d[id_state][id_visibility]=read_model(folder, model, id_network, id_measurement, id_state, id_visibility)
            except:
                print("no model_{}_{}_{}_{}".format(id_network, id_measurement, id_state, id_visibility))
                d[id_state][id_visibility]=0
    return d
    
def read_all_cs2(folder, id_measurement, id_network=0, width=64, depth=3, no_checkpoint=False, n_sample=50000, dist = "kl"):
    d = np.zeros([21,3])
    
    for id_state in range(21):
        # for id_state in [6,]:
        for id_visibility in [0,1,2]:
            config,_,_ = get_network(id_network)
            model = NeuralNetwork(config, width, num_hidden_layers=depth, dist=dist)
            try:
                if no_checkpoint:
                    target_distribution = get_precomputed_distr("changing_states2_{}_{}/distributions_{}_{}.txt".format(id_network, id_measurement, id_state, id_visibility))
                    d[id_state][id_visibility]=read_model(folder, model, id_network, id_measurement, id_state, id_visibility, no_checkpoint, target_distribution, n_sample)
                else:
                    d[id_state][id_visibility]=read_model(folder, model, id_network, id_measurement, id_state, id_visibility)
            except:
                print("no model_{}_{}_{}_{}".format(id_network, id_measurement, id_state, id_visibility))
                d[id_state][id_visibility]=0
    return d

def read_all_ws(folder, width=60, depth=4, no_checkpoint=False, n_sample=50000, dist = "kl"):
    d = np.zeros([2,6,8,3])
    for id_network in [0,1]:
        for id_measurement in [0,1,2,3,4,5]:
            for id_state in [0,1,2,3,4,5,6,7]:
                for id_visibility in [0,1,2]:
                    config,_,_ = get_network(id_network)
                    model = NeuralNetwork(config, width, num_hidden_layers=depth, dist=dist)
                    # try:
                    if no_checkpoint:
                        target_distribution = get_precomputed_distr("all_elegant_d/distributions_{}_{}_{}_{}.txt".format(id_network, id_measurement, id_state, id_visibility))
                        d[id_network][id_measurement][id_state][id_visibility]=read_model(folder, model, id_network, id_measurement, id_state, id_visibility, no_checkpoint, target_distribution, n_sample)
                    else:
                        d[id_network][id_measurement][id_state][id_visibility]=read_model(folder, model, id_network, id_measurement, id_state, id_visibility).detach().numpy()
            # except:
            #     print("no model_{}_{}_{}_{}".format(id_network, id_measurement, id_state, id_visibility))
            #     d[id_network][id_measurement][id_visibility]=0
    return d

        
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
    
    
def read_model_from_path(folder, id_network, id_measurement, id_state, id_visibility, width=64, depth=3):
    path ='model/{}/model_{}_{}_{}_{}_opt3.pth'.format(folder, id_network, id_measurement, id_state, id_visibility)
    config,_,_ = get_network(id_network)
    model = NeuralNetwork(config, width, num_hidden_layers=depth)
    return get_dist_from_model(model, path).detach().numpy()
    # load_model_and_optimizer(model, path)
                   

        
    
def get_distr_from_model(path, id_network, width, depth):
    """return the distribution of a local model"""
    config,_,_ = get_network(id_network)
    model = NeuralNetwork(config, width, num_hidden_layers=depth)
    target_distribution = get_precomputed_distr(path)
    distr=evaluate_model_return_distribution(model, 50000, target_distribution, opt=3, path=path)
    return distr


def read_one(folder, model, i, no_checkpoint=False, target_distribution=None, n_sample = 50000):
        save_path = '{}/model_{}_opt3.pth'.format(folder, i)
        if no_checkpoint:
            return evaluate_model_return_dist(model, save_path, n_sample, target_distribution, 3)
        else:
            return get_dist_from_model(model, save_path)
  
    
def read(folder, n_params, id_network, width=60, depth=4, no_checkpoint=False, folder_target=None, n_sample=50000, dist = "kl"):
    """general function to read a folder sweeping one parameter only"""
    d = np.zeros([n_params,])
    
    config,_,_ = get_network(id_network)
    model = NeuralNetwork(config, width, num_hidden_layers=depth, dist=dist)
    # try:
    for i in range(n_params):
        if no_checkpoint:
            target_distribution = get_precomputed_distr("{}/distributions_{}.txt".format(folder_target, i))
            d[i]=read_one(folder, model, i, no_checkpoint, target_distribution, n_sample)
        else:
            d[i]=read_one(folder, model, i)
    return d

def plot_elegant_triangle(dist, no_checkpoint=False, n_sample=50000, folder_target=None):
    if dist=="eucl":
        d = read("model/elegant/elegant_eucl", 21, 0, no_checkpoint=no_checkpoint, n_sample=n_sample, folder_target=folder_target, dist = dist)
        plot_simple(d, 0.5,1, title="elegant distribution", xlabel="visibility", ylabel="euclidean distance")
    elif dist=="kl":
        d = read("model/elegant/elegant_kl", 21, 0, no_checkpoint=no_checkpoint, n_sample=n_sample, folder_target=folder_target, dist = dist)
        plot_simple(d, 0.5,1, title="elegant distribution", xlabel="visibility", ylabel="KL divergence")
    return d

    
def plot_rgb4_triangle(dist_loc, u_or_v, dist_read=None, no_checkpoint=False, n_sample=50000):
    """read results for rgb4 distr. dist_loc is for the name of the folders, dist_read is the actual distance used to 
    evaluate the results """
    if dist_read==None:
        dist_read =dist_loc
    folder_target = "target_distributions/RGB4_"+u_or_v
    if dist_loc=="eucl":
        if u_or_v=="u":
            d = read("model/rgb4/rgb4u_eucl", 21, 0, no_checkpoint=no_checkpoint, n_sample=n_sample, folder_target=folder_target, dist = dist_read)
            plot_simple(d, 0.5,1, title="RBG4 distribution", xlabel=r"$u^2$", ylabel="euclidean distance", ymin=0)
        elif u_or_v=="v":
            d = read("model/rgb4/rgb4v_eucl", 21, 0, no_checkpoint=no_checkpoint, n_sample=n_sample, folder_target=folder_target, dist = dist_read)
            plot_simple(d, 0.7,1, title="RBG4 distribution", xlabel="visibility", ylabel="euclidean distance", ymin=0)
        elif u_or_v=="vz":
            d = read("model/rgb4/rgb4v_z_eucl", 21, 0, no_checkpoint=no_checkpoint, n_sample=n_sample, folder_target=folder_target, dist = dist_read)
            plot_simple(d, 0.95,1, title="RBG4 distribution", xlabel="visibility", ylabel="euclidean distance", ymin=0)

    elif dist_loc=="kl":
        if u_or_v=="u":
            d = read("model/rgb4/rgb4u", 21, 0, no_checkpoint=no_checkpoint, n_sample=n_sample, folder_target=folder_target, dist = dist_read)
            plot_simple(d, 0.5,1, title="RBG4 distribution", xlabel=r"$u^2$", ylabel="KL divergence", ymin=0)
        elif u_or_v=="v":
            d = read("model/rgb4/rgb4v", 21, 0, no_checkpoint=no_checkpoint, n_sample=n_sample, folder_target=folder_target, dist = dist_read)
            plot_simple(d, 0.7,1, title="RBG4 distribution", xlabel="visibility", ylabel="KL divergence", ymin=0)
        elif u_or_v=="vz":
            d = read("model/rgb4/rgb4v_z", 21, 0, no_checkpoint=no_checkpoint, n_sample=n_sample, folder_target=folder_target, dist = dist_read)
            plot_simple(d, 0.95,1, title="RBG4 distribution", xlabel="visibility", ylabel="KL divergence", ymin=0)
    return d

def plot_2d_map(d, vmax=None):
    if vmax is not None:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        colormap=plt.imshow(d, norm=norm)
    else:
        colormap=plt.imshow(d)

    xticks=['0', r'$\pi$/2', r'$\pi$']
    plt.xticks(ticks=[0,5,10], labels=xticks)
    
    yticks=['0', r'$\pi/4$', r'$\pi/2$']
    plt.yticks(ticks=[0,5,10], labels=yticks)
    
    plt.ylabel("angle measurement")
    plt.xlabel("angle state")
    
    plt.colorbar(colormap, label="KL divergence")
    plt.show()
    
def read_sweep_2d(family_state=1, id_network=0, width=60, depth=4, no_checkpoint=False, folder_target=None, n_sample=50000, dist = "kl", dist_max=None):
    """function to read the 2d plots that vary the measurement and the state"""
    family_state_label = "" if family_state==1 else "2"
    folder = "sweep/sweep_ms{}_{}_{}".format(family_state_label, id_network, dist)
    d = np.zeros([11,11])
    id_visibility=0
    config,_,_ = get_network(id_network)
    model = NeuralNetwork(config, width, num_hidden_layers=depth, dist=dist)
    # try:
    for id_meas in range(11):
        for id_state in range(11):
            d[id_meas][id_state]=read_model(folder, model, id_network, id_meas, id_state, id_visibility)
    
    plot_2d_map(d, dist_max)
    return d

if __name__=="__main__":
    ## plot the results in the rgb4 triangle
    ## first argument is the distance: "kl" or "eucl"
    ## second argument chooses the variable: "u" for the parameter in the measurement, "v" for the visibility, "vz" for a zoom on the visibility
    ## dist_read set the type of distance to read the results if we want to read the results with another distance than the training one
    ## no_checkpoint= False to get the result during training, True for evaluating the model with n_sample as the number of samples

    d=plot_rgb4_triangle("eucl", "vz", dist_read = None, no_checkpoint=False, n_sample=1000000)
    
    ## plot the results in the elegant triangle
    ## the first argument is the distance: "kl" or "eucl"
    
    # d=plot_elegant_triangle("kl", no_checkpoint=False, n_sample=100000, folder_target="target_distributions/elegant_v")
    
    ## plot the 2d plot changing the measurement and the state.
    ## family_state is 1 for the state going from phi+ to psi+ and 2 for the state going from phi- to psi-
    ## the id_network available are 0:triangle, 1:square, 4:pentagon, 6:pentagon with 3 outputs
    ## only the distance kl is available, except for id_network=6
    ## dist_max set the max distance for the color scale
    
    # d=read_sweep_2d(family_state=1, id_network= 6, dist = "kl", dist_max=None)

