import sys
sys.path.append('../')

from module.nn_exact_strategies.build_nn import NeuralNetwork, train_model_and_save, load_model_and_optimizer
import numpy as np
import shutil  
from utils.networks import get_network
import os
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device used :", device)

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

#######################################################
"""
file to run the Neural Network on a target distribution
see the main part below to choose parameters
"""
def run_NN(config, target_distribution, cardinality, folder ='model/temp', name_model = "model", training_dist = "kl",
           n_training_steps = 40000, start_from_other_path=None, 
           retrain=False, width=60, depth=4, n_retries=0, threshold=0):
    
    model = NeuralNetwork(config, target_distribution, cardinality, width, depth=depth, dist=training_dist)
    
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
        load_model_and_optimizer(model, start_from_other_path)
        model.best_dist = float('inf') #to make sure we start saving from the start at the new path
        model.dist=training_dist

    
    # first round training
    if not retrain:
        train_model_and_save(model, n_training_steps, model_path=save_path, threshold=threshold)
    # retraining
    for i in range(n_retries):
        load_model_and_optimizer(model, save_path)
        best_dist = model.best_dist
        del model
        model = NeuralNetwork(config, target_distribution, cardinality, width, depth=depth, dist=training_dist)
        if best_dist > threshold :
            if start_from_other_path is not None:
                load_model_and_optimizer(model, start_from_other_path)
                model.best_dist = float('inf') #to make sure we start saving from the start at the new path
                model.dist=training_dist
            
            train_model_and_save(model, n_training_steps, model_path=save_path2, threshold=threshold)

            load_model_and_optimizer(model, save_path2)
            best_dist2 = model.best_dist
            if best_dist2 < best_dist:
                print("new model is better, saving in main folder")
                shutil.copyfile(save_path2, save_path)
        else:
            break


def get_target_distribution(path):
    f=open(path)
    out = np.loadtxt(f)
    f.close()
    return out

if __name__=="__main__":
    # # to set up parameters running from command lines
    # import sys
    # if len(sys.argv)==2:
    #     id_visibility=int(sys.argv[1])
    # elif len(sys.argv)==0:
    #     pass #no parameters
    # else:
    #     print("error with sys.argv: len(sys.argv)= ", len(sys.argv))
    
    relative_path = "../" 
    id_distribution = 0
    cardinality = 5
    id_network = 5
    project_name = "binary_triangle/test_binary_distr"

    training_dist = "eucl" #'kl' for Kullback–Leibler, 'eucl' for euclidean. Define custom distance in module/neural_network_module_pytorch/utils/torch_probabilites.py
    n_training_steps = 100
    folder = relative_path+"model/exact_strategies/"+project_name+"_card"+str(cardinality)+"eucl"
    # folder = "model/test/model_test_card"+str(cardinality)

    name_model = "model_"+str(id_distribution)
    width=60
    depth=4
    
    n_retries=5 #number of times we retrain the target. 0 for training only once
    start_from_other_path = relative_path+"model/exact_strategies/"+project_name+"_card"+str(cardinality)+"/model_0.pth" #"model/" #write the path to start from this folder instead
    threshold = 0.0001 #distance at which we stop training
    
    retrain=False

    ## choose topology of the network (see utils/network for some predefined networks)
    ## 0: triangle, 1:square, 2:square with a diagonal, 3:square with two diagonal, 4:pentagon
    
    config = get_network(id_network) 
    
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
    target_distribution_path = relative_path+"/target_distributions/{}/distribution_{}.txt".format(project_name, id_distribution)
    target_distribution=get_target_distribution(target_distribution_path)
    
    # model = NeuralNetwork(config, width, cardinality, depth=depth, dist=training_dist)

    ## main function defined above to control the training
    run_NN(config, target_distribution, cardinality, folder=folder, name_model=name_model, 
                    training_dist = training_dist, n_training_steps = n_training_steps,
                    start_from_other_path=start_from_other_path, retrain=retrain, width=width, 
                    depth=depth, n_retries = n_retries, threshold=threshold)

