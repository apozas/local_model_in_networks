from module.neural_network_module_pytorch.build_nn import NeuralNetwork, train_model_and_save, load_model_and_optimizer, evaluate_model, load_model_without_checkpoints, evaluate_model_return_dist
import numpy as np
import shutil  
from networks import get_network
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
           retrain=False, width=60, depth=4, n_retries=0, threshold=0, bias=4, n_steps_reevaluate=1000, min_sampling=1000, max_sampling=1000000, opt=1):
    
    model = NeuralNetwork(config, target_distribution, width, depth=depth, dist=training_dist, n_samples=n_samples, opt=opt)
    
    folder2 = '{}/retrain'.format(folder)
    save_path = '{}/{}.pth'.format(folder, name_model)
    save_path2 = '{}/{}.pth'.format(folder2, name_model)

    # first round training
    if not retrain:
        ## choose starting point
        if start_from_other_path is not None:
            load_model_without_checkpoints(model, max_sampling, start_from_other_path, dist = training_dist)
            model.n_samples = max_sampling
            model.best_dist = float('inf') #to make sure we start saving from the start at the new path
        
        train_model_and_save(model, n_samples, n_training_steps, 
                             model_path=save_path, threshold=threshold, bias=bias,
                             n_steps_reevaluate=n_steps_reevaluate, min_sampling=min_sampling,
                             max_sampling=max_sampling)
    # retraining
    for i in range(n_retries):
        load_model_and_optimizer(model, save_path)        
        best_dist = model.best_dist
        del model
        model = NeuralNetwork(config, target_distribution, width, depth=depth, dist=training_dist, n_samples=n_samples, opt=opt)
        if best_dist > threshold :
            ## choose starting point
            if start_from_other_path is not None:
                load_model_without_checkpoints(model, max_sampling, start_from_other_path, dist = training_dist)
                model.n_samples = max_sampling
                model.best_dist = float('inf') #to make sure we start saving from the start at the new path
                
            train_model_and_save(model, n_samples, n_training_steps, 
                                 model_path=save_path2, threshold=threshold, bias=bias,
                                 n_steps_reevaluate=n_steps_reevaluate, min_sampling=min_sampling,
                                 max_sampling=max_sampling)
            load_model_and_optimizer(model, save_path2)
            best_dist2 = model.best_dist
            if best_dist2 < best_dist:
                print("new model is better, saving in main folder")
                shutil.copyfile(save_path2, save_path)
        else:
            print("Not training: current distance at ", best_dist)
            break


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
    n_training_steps = 10000
    folder = "model/test/"
    name_model = "model_test"
    width=60
    depth=4
    opt=1 #an option to sample the local variable (input of the neural network)
    n_retries=3 #number of times we retrain the target. 0 for training only once
    start_from_other_path = None #"model/test/model_test.pth" #path to start from (parameters of this model will replace the given ones)
    threshold = 1e-6 #distance at which we stop training
    n_steps_reevaluate = 10000 #after how many steps we reevaluate the best model (to avoid being stuck because of a lucky evaluation)
    
    n_samples = None #None for adapting samples, choose an integer for a fixed number of samples
    retrain= False #to improve an existing result and keeping the previous one if better. If False, the new model will remplace the model at the same address if it exists
    
    ## if n_samples is None, we can control the min and max value (ignored if n_samples is not None)
    max_sampling = 80000 #sampling value too large can cause out of memory errors
    min_sampling = 5000 #sampling too low can lead to slow training  
    bias = 4 #for the sampling, more means more sampling (relatif to current loss)
    
    ## choose topology of the network
    ## 0: triangle, 1:square, 2:square with a diagonal, 3:square with two diagonal, 4:pentagon, 5:binary triangle, 6:pentagon with three outputs, 7:tetrahedron
    
    config = get_network(0) 
    
    ## or build one
    ## 'a', 'b', 'c' are the parties, the grec letter are the sources these parties have access to. 
    ## The number is the number of outcomes of this party
    
    # config = {
    #     'a': (['beta', 'gamma'], 4),
    #     'b': (['alpha', 'gamma'], 4),
    #     'c': (['alpha', 'beta'], 4)
    # }

    ## get the target distribution (as an np array). Array should be one dimensional with outcomes in order p_000, p_001, ...
    target_distribution_path = "target_distributions/test/rgb4_umax.txt"
    # target_distribution_path = "target_distributions/tetrahedron/elegant_distr_tetra.txt"


    target_distribution=get_target_distribution(target_distribution_path)

    ## main function defined above to control the training
    run_NN(config, target_distribution, folder=folder, name_model=name_model, 
                   training_dist = training_dist, n_training_steps = n_training_steps,
                   start_from_other_path=start_from_other_path, n_samples=n_samples,
                   retrain=retrain, width=width, depth=depth, n_retries = n_retries, threshold=threshold, bias = bias,
                   n_steps_reevaluate=n_steps_reevaluate, min_sampling=min_sampling, max_sampling=max_sampling, opt=opt)

