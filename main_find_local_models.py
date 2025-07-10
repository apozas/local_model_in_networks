from module.neural_network_module_pytorch.build_nn import NeuralNetwork, train_model_and_save, load_model_and_optimizer, evaluate_model, load_model_without_checkpoints, evaluate_model_return_dist
import numpy as np
import shutil  
from networks import get_network
import os
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device used :", device)

#######################################################
"""
file to run the Neural Network on a target distribution
see the main part below to choose parameters
"""
def run_NN(config, target_distribution, folder ='model/temp', name_model = "model", training_dist = "kl",
           n_training_steps = 40000, n_samples=None, start_from_other_path=None, 
           retrain=False, width=60, depth=4, n_retries=0, threshold=0, biais=4, n_epochs_reevaluate=1000, max_sampling=1000000):
    
    model = NeuralNetwork(config, width, num_hidden_layers=depth, dist=training_dist)
    
    folder2 = '{}/retrain'.format(folder)
    save_path = '{}/{}.pth'.format(folder, name_model)
    save_path2 = '{}/{}.pth'.format(folder2, name_model)

    ## check current best model
    if retrain:
        load_model_and_optimizer(model, save_path)
        best_dist = model.best_dist
        if best_dist<threshold:
            print("Not training: current distance at ", best_dist)
            return 0
    
    ## choose starting point
    if start_from_other_path is not None:
        # load_model_without_checkpoints(model, 100000, target_distribution, 3, start_from_path, dist = "eucl")
        load_model_and_optimizer(model, start_from_other_path,  dist = "kl")
        model.best_dist = float('inf') #to make sure we start saving from the start at the new path
    
    # first round training
    if not retrain:
        train_model_and_save(model, n_samples, target_distribution, 
                              n_training_steps, model_path=save_path, value_disp=1000000, 
                              opt=3, threshold=threshold, 
                              n_epochs_reevaluate=n_epochs_reevaluate, biais=biais, max_sampling=max_sampling)

    # retraining
    for i in range(n_retries):
        load_model_and_optimizer(model, save_path)
        best_dist = model.best_dist
        del model
        model = NeuralNetwork(config, width, num_hidden_layers=depth)
        if best_dist > threshold :
            train_model_and_save(model, n_samples, target_distribution, 
                                 n_training_steps, model_path=save_path2, value_disp=1000000, 
                                 opt=3, threshold=threshold, 
                                 n_epochs_reevaluate=n_epochs_reevaluate, biais=biais, max_sampling=max_sampling)
            load_model_and_optimizer(model, save_path2)
            best_dist2 = model.best_dist
            if best_dist2 < best_dist:
                print("new model is better, saving in main folder")
                shutil.copyfile(save_path2, save_path)
        else:
            break


def get_elegant_distr(id_network, id_measurement, id_state, id_visibility):
    if id_network == 0:
        f= open("all_elegant_d/distributions_{}_{}_{}_{}.txt".format(id_network, id_measurement, id_state, id_visibility), 'r')
    elif id_network == 1:
        f= open("all_eBSM_square_d/distributions_{}_{}_{}_{}.txt".format(id_network, id_measurement, id_state, id_visibility), 'r')
    out = np.loadtxt(f)
    f.close()
    return out

def get_target_distribution(path):
    f=open(path)
    out = np.loadtxt(f)
    f.close()
    return out

if __name__=="__main__":
    ## to set up parameters running from command lines
    # import sys
    # if len(sys.argv)==5:
    #     id_network=int(sys.argv[1])
    #     id_measurement=int(sys.argv[2])
    #     id_state=int(sys.argv[3])
    #     id_visibility=int(sys.argv[4])
    # elif len(sys.argv)==0:
    #     pass #no parameters
    # else:
    #     print("error with sys.argv: len(sys.argv)= ", len(sys.argv))
        
    
    training_dist = "kl" #'kl' for Kullback–Leibler, 'eucl' for euclidean. Define custom distance in module/neural_network_module_pytorch/utils/torch_probabilites.py
    n_training_steps = 100000
    folder = "model/test"
    name_model = "model_test"
    width=60
    depth=4
    n_retries=5 #number of times we retrain the target. 0 for training only once
    start_from_other_path = None #"model/" #write the path to start from this folder instead
    threshold = 0.0001 #distance at which we stop training
    biais = 4 #for the sampling, more means more sampling
    n_epochs_reevaluate = 5000 #after how many steps we reevaluate the best model (to avoid being stuck because of a lucky evaluation)
    
    n_samples = None #None for adapting samples
    
    ## if n_samples is None, we can control the min and max value (ignored if n_samples is not None)
    max_sampling = 1000000 #sampling value too large can cause out of memory errors
    min_sampling = 1000 #sampling too low can lead to slow training  
    biais = 4 #for the sampling, more means more sampling (relatif to current loss)
    
    ## choose topology of the network
    ## 0: triangle, 1:square, 2:square with a diagonal, 3:square with two diagonal, 4:pentagon
    
    config, _, _ = get_network(0) 
    
    ## or build one
    ## 'a', 'b', 'c' are the parties, the grec letter are the sources these parties have access to. 
    ## The number is the number of outcomes of this party
    
    # config = {
    #     'a': (['beta', 'gamma'], 4),
    #     'b': (['alpha', 'gamma'], 4),
    #     'c': (['alpha', 'beta'], 4)
    # }

    # target_distribution=get_elegant_distr(id_network, id_measurement, id_state, id_visibility)
    ## get the target distribution (as an np array). Array should be one dimensional with outcomes in order p_000, p_001, ...
    target_distribution_path = "target_distributions/test/rgb4_umax.txt"
    target_distribution=get_target_distribution(target_distribution_path)

    ## main function defined above to control the training
    run_NN(config, target_distribution, folder=folder, name_model=name_model, 
                   training_dist = training_dist, n_training_steps = n_training_steps,
                   start_from_other_path=start_from_other_path, n_samples=n_samples,
                   retrain=False, width=width, depth=depth, n_retries = n_retries, threshold=threshold, biais = biais,
                   n_epochs_reevaluate=n_epochs_reevaluate, max_sampling=max_sampling)

