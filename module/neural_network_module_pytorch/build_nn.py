from module.neural_network_module_pytorch.utils.torch_probabilities import CustomLoss, custom_loss_distribution
# import torch.nn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########################Build Model##################################


class NeuralNetwork(nn.Module):
    def __init__(self, config, layers, num_hidden_layers=3, lr=0.001, dist="kl"):
        super(NeuralNetwork, self).__init__()
        self.input_layers = nn.ModuleDict()
        self.hidden_layers = nn.ModuleDict()
        self.output_layers = nn.ModuleDict()
        self.config = config
        self.best_dist = float('inf')
        self.dist = dist #type of distance, should be "kl" or "eucl". Define other distance in torch_probabilities.py

        sources = set(src for party in config for src in config[party][0])
        self.source_indices = {source: idx for idx, source in enumerate(sources)}

        for party, (sources, value) in config.items():
            input_dim = len(sources)
            hidden_layer_size = layers * len(sources)
            
            self.input_layers[party] = nn.Linear(input_dim, hidden_layer_size)
            
            hidden_layers = []
            for _ in range(num_hidden_layers):
                hidden_layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))
                hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Linear(hidden_layer_size, value))
            
            self.hidden_layers[party] = nn.Sequential(*hidden_layers)
            self.output_layers[party] = nn.Softmax(dim=1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        outputs = []
        for party, layer in self.input_layers.items():
            input_indices = [self.source_indices[source] for source in self.config[party][0]]
            selected_inputs = x[:, input_indices]
            h = layer(selected_inputs)
            h = self.hidden_layers[party](h)
            out = self.output_layers[party](h)
            outputs.append(out)

        joint_prob = torch.cat(outputs, dim=1)
        return joint_prob
    
    def outputsize(self):
        keys= list(self.config.keys())
        n_outputs = 0
        for akey in keys:
            n_outputs+= self.config[akey][1]
        return n_outputs
    
    def initialize_weights(self):
        for m in self.modules():
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            # if isinstance(m, nn.Conv2d):
            #     nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    
            #     if m.bias is not None:
            #         nn.init.constant_(m.bias, 0)
    
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)
    
            # elif isinstance(m, nn.Linear):
            #     nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    
            #     if m.bias is not None:
            #         nn.init.constant_(m.bias, 0)
            
        




#############################Generate Data#####################################



def generate_data(config, n_value_total, target_distribution, opt=3):
    """generate the input for the neural network, different distribution available"""
    data_type = np.float32 
    
    sources = list(set(source for person in config for source in config[person][0]))
    n_sources = len(sources)
    
    
    if opt == 1:
        data = np.random.randn(n_value_total, n_sources).astype(data_type)
        data = (data - data.min()) / (data.max() - data.min())
    elif opt == 2:
        data = np.random.randn(n_value_total, n_sources)
        data = np.divide((data - 0.5), 0.28867513459).astype(data_type)
    elif opt == 3:
        data = np.random.randn(n_value_total, n_sources)
        data = np.divide((data - 0.5), 0.1443375673).astype(data_type)
    elif opt == 4:
        data = np.random.uniform(-20, 20, (n_value_total, n_sources)).astype(data_type)
    elif opt == 5:
        data = np.random.randn(n_value_total, n_sources).astype(data_type)
        data = 1 / (1 + np.exp(-data))
    elif opt ==6 :
        data = np.random.uniform(0, 1, (n_value_total, n_sources)).astype(data_type)
    else:
        raise ValueError("Invalid option for 'opt'. Use 1 for normalized standard normal distribution, 2 for adjusted distribution with mean -0.5 and variance 0.28867513459, 3 for adjusted distribution with mean -0.5 and variance 0.1443375673, or 4 for uniform distribution between -20 and 20.")
    
    labels = np.tile(target_distribution, (n_value_total, 1)).astype(data_type)
    
    return data, labels


################################Train Model########################


def sampling(loss, n_output, dist, biais=10, min_sampling=1000, max_sampling=float("inf")):
    """return the optimal number for sampling for a given loss and number of outputs"""
    ## found by plotting kl divergence of sampled distribution for 64 outputs
    ## the sampling is linear with the number of outputs
    ## the biais tells how precise the distribution should be 
    def func_kl(x):
        A=1
        B=2
        return 1/(B*x**A)
    def func_eucl(x):
        A=2
        B=1
        return 1/(B*x**A)
    if dist=="kl" or "alex":
        out = min(max(int(func_kl(loss/biais)*n_output), min_sampling), max_sampling)
    elif dist=="eucl":
        out = min(max(int(func_eucl(loss/biais)), min_sampling), max_sampling)
    return out

def train_model_and_save(model, n_value_total_in, target_distribution, epochs, model_path='model/model.pth', value_disp=10, opt=1, threshold=0, biais=10, compile_model=False, n_epochs_reevaluate=10000, min_sampling = 1000, max_sampling=float("inf"), biais_max=10, value_empty_cache=float("inf")):
    """main function for training.
    min_sampling, max_sampling and biais_max control how much we sample during training if n_value_total_in is None"""
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    if compile_model:
        model = torch.compile(model)
    model.to(device)
    
    
    criterion = CustomLoss(model.config, dist=model.dist)
    np.set_printoptions(precision=8, suppress=False)
    
    #fonction to generate the distribution from torch_probabilities, compiled for speed (no improvement)
    compiled_distribution_function = torch.compile(custom_loss_distribution)
    
    #initiate number of sampling, can change during training
    if n_value_total_in is None:
        n_value_total = 5000
    else:
        n_value_total = n_value_total_in
        
    n_outputs = len(target_distribution)
    
    n_epochs_no_update = 0
    
    with tqdm(range(epochs)) as t:
        for epoch in t:
            
            model.train()
    
            data, labels = generate_data(model.config, n_value_total, target_distribution, opt)
            # data, labels = torch.tensor(data).clone().detach().to(device), torch.tensor(labels).clone().detach().to(device)
            data, labels = torch.tensor(data).to(device), torch.tensor(labels).to(device)
    
            total_outputs = model(data)
            model.optimizer.zero_grad()
            loss, predict= criterion(total_outputs, labels, compiled_distribution_function)
            
            loss.backward()
            model.optimizer.step()
            
            ## automatic change of the sampling (n_value_total) with the loss
            
            if n_value_total_in is None:
                n_value_total= sampling(loss, n_outputs, model.dist, biais, min_sampling, max_sampling)
                
                
            if (epoch % value_disp == value_disp - 1):
                print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
                predict_numpy = predict.detach().cpu().numpy()
                print(f"Predicted probability:\n{predict_numpy}")
                
            if (epoch % value_empty_cache == 0):
                print(f"Epoch {epoch}, emptying cache")
                torch.cuda.empty_cache()
                

            if loss < model.best_dist:
                model.best_dist = loss
                
                elapsed = t.format_dict['elapsed']
                elapsed_str = t.format_interval(elapsed)
    
                print(f"update model : dist ({model.dist}) =", np.around(float(loss), 5), "time ", elapsed_str, "n_value_total", n_value_total)
                save_model_and_optimizer(model, model_path)
                n_epochs_no_update = 0
                
            else:
                ## reevaluation of the best model
                n_epochs_no_update += 1
                if n_epochs_no_update > n_epochs_reevaluate:
                    model.best_dist = evaluate_model_return_dist(model, model_path, n_value_total*5, target_distribution, opt, model.dist)
                    n_epochs_no_update = 0
                    
                    if biais<biais_max:
                        biais+=1 #increase the sampling to end the training and making sure we are not under sampling
                    print("reevaluation of the model : {}".format(model.best_dist))
                        
            if loss<threshold:
                print("stopping the training: reached the threshold of {}".format(threshold))
                break

    try:
        print(f"Loss ({model.dist}): {model.best_dist}\n ")
    except:
        print("No best result found")
        
    return model.best_dist



####################################Save and Load################################

def save_model_and_optimizer(model, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'best_dist': model.best_dist,
    }, path)



def load_model_and_optimizer(model, path='model/model.pth', dist='kl'):
    model.to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.best_dist = checkpoint.get('best_dist', float('inf'))
    print("Modèle et optimiseur chargés depuis :", path)
    print("Meilleure distance {} enregistrée :".format(dist), model.best_dist)

def load_model_without_checkpoints(model, n_value_total, target_distribution, opt, path='model/model.pth', dist ='kl'):
    model.to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.best_dist = evaluate_model_return_dist(model, n_value_total, target_distribution, opt, dist =dist)
    
    print("Modèle et optimiseur chargés depuis :", path)
    print("Meilleure distance {} enregistrée :".format(dist), model.best_dist)

    
def get_dist_from_model(model, path='model/model.pth'):
    """return the distance of a saved model
    inputs : model (configuration of NN), path of saved model
    outputs : distance"""
    checkpoint = torch.load(path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.best_dist = checkpoint.get('best_dist', float('inf'))

    return model.best_dist


####################################Evaluate######################################


def evaluate_model(model, n_value_total, target_distribution, filename, opt=1):
    criterion = CustomLoss(model.config, dist=model.dist)
    model.eval() 
    
    data, labels = generate_data(model.config, n_value_total, target_distribution, opt)

    total_outputs = model(data)

    d, predict = criterion(total_outputs, labels)
    
    # Écrire les résultats dans un fichier, créer le fichier s'il n'existe pas
    with open(filename, 'w') as file:
        file.write(f"{model.dist} distance: {d}\n")
        file.write(f"Predicted probability:\n{predict.detach().cpu().numpy()}\n")

    print(f"Results saved to {filename}")

def evaluate_model_return_distribution(model, n_value_total, target_distribution, opt=1, path="model/model.pth"):
    checkpoint = torch.load(path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    criterion = CustomLoss(model.config, dist=model.dist)
    model.eval() 
    
    data, labels = generate_data(model.config, n_value_total, target_distribution, opt)
    data, labels = torch.tensor(data).to(device), torch.tensor(labels).to(device)

    total_outputs = model(data)

    d, predict = criterion(total_outputs, labels)
    
    return predict.detach().cpu().numpy()

    
def evaluate_model_return_dist(model, path, n_value_total, target_distribution, opt=1, dist=None):
    if dist is None:
        dist = model.dist
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    criterion = CustomLoss(model.config, dist)
    model.eval() 
    total_probs = []

    data, labels = generate_data(model.config, n_value_total, target_distribution, opt)
    data, labels = torch.tensor(data).to(device), torch.tensor(labels).to(device)

    total_outputs = model(data)
    d, predict = criterion(total_outputs, labels)
    return d



def output_model(model, n_value_total, target_distribution, opt=1):
    model.eval()
    
    data, _ = generate_data(model.config, n_value_total, target_distribution, opt)

    # data = np.array([[0.2, 0.3, 0.4]], dtype=np.float32)

    data = torch.tensor(data).to(device)

    with torch.no_grad(): 
        outputs = model(data)


    separated_outputs = []
    start_idx = 0
    for party, (_, output_size) in model.config.items():
        end_idx = start_idx + output_size
        separated_outputs.append(outputs[:, start_idx:end_idx])
        start_idx = end_idx


    grouped_outputs = torch.stack(separated_outputs, dim=1)

    return data, grouped_outputs

def plot_distribution(model, path, n_value_total, target_distribution, opt=1, dist='kl'):
    import matplotlib.pyplot as plt
        
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    criterion = CustomLoss(model.config, dist)
    model.eval() 
    total_probs = []

    data, labels = generate_data(model.config, n_value_total, target_distribution, opt)
    data, labels = torch.tensor(data).to(device), torch.tensor(labels).to(device)

    total_outputs = model(data)
    d, predict = criterion(total_outputs, labels)
    
    
    plt.scatter(np.arange(len(predict)), predict.detach().numpy())
    plt.scatter(np.arange(len(target_distribution)), target_distribution)
    plt.title(path + "  distance = {}".format('{0:.6f}'.format(d)))
    plt.plot



    
    
    





