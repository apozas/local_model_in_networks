from module.nn_exact_strategies.utils.torch_probabilities import CustomLoss, custom_loss_distribution
# import torch.nn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np
from tqdm import tqdm
from itertools import product
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########################Build Model##################################


class NeuralNetwork(nn.Module):
    def __init__(self, config={}, target_distribution=[], cardinality=6, width=60, depth=4, lr=0.001, dist="kl", scale_blocks_with_n_sources=False):
        super(NeuralNetwork, self).__init__()
        self.config = config
        self.cardinality = cardinality
        self.target_distribution = target_distribution
        self.width=width
        self.depth=depth
        self.lr=lr
        self.dist = dist #type of distance, should be "kl" or "eucl". Define other distance in torch_probabilities.py
        self.scale_blocks_with_n_sources = scale_blocks_with_n_sources #option to have the block of a party with width multiplied by the number of sources connected


        self.input_sources_width = nn.ModuleDict()
        self.input_party_width = nn.ModuleDict()
        self.hidden_width = nn.ModuleDict()
        self.output_width = nn.ModuleDict()
        self.best_dist = float('inf')
        self.all_symbols=dict()
        
        self.sources = set(src for party in self.config for src in self.config[party][0])
        self.source_indices = {source: idx for idx, source in enumerate(self.sources)}
        
        try:
            self.build_model()
        except:
            pass
        
    def build_model(self):
        self.input_sources_width = nn.ModuleDict()
        self.input_party_width = nn.ModuleDict()
        self.hidden_width = nn.ModuleDict()
        self.output_width = nn.ModuleDict()
        
        self.sources = set(src for party in self.config for src in self.config[party][0])
        self.source_indices = {source: idx for idx, source in enumerate(self.sources)}

        
        for source in self.sources:
            input_dim = 1
            self.input_sources_width[source] = nn.Linear(input_dim, self.cardinality)
            self.output_width[source] = nn.Softmax(dim=1)
            
        for party, (sources, value) in self.config.items():
            input_dim = len(sources)*self.cardinality
            
            if self.scale_blocks_with_n_sources:
                hidden_layer_size = self.width * len(self.sources)
            else:
                hidden_layer_size = self.width
            
            self.input_party_width[party] = nn.Linear(input_dim, hidden_layer_size)
            
            hidden_width = []
            for _ in range(self.depth):
                hidden_width.append(nn.Linear(hidden_layer_size, hidden_layer_size))
                hidden_width.append(nn.ReLU())
            hidden_width.append(nn.Linear(hidden_layer_size, self.cardinality**len(sources)*value))
            
            self.hidden_width[party] = nn.Sequential(*hidden_width)
            
            symbols = ""
            for i in range(self.cardinality):
                symbols+=str(i)
            self.all_symbols[party]=symbols
            for received_symbols in product(symbols, repeat=len(sources)):
                name = party
                for n_source in range(len(sources)):
                    name=name+"_"+sources[n_source]+received_symbols[n_source]          
                self.output_width[name]= nn.Softmax(dim=1)
                
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        outputs_sources = []
        outputs = []
        
        for source, layer in self.input_sources_width.items():
            # input_indices = self.source_indices[source]
            selected_inputs = x
            selected_inputs = selected_inputs.reshape([1, 1])
            h = layer(x)
            h = h.reshape([1, len(h)])
            out_sources = self.output_width[source](h)
            
            # out_sources = out_sources.reshape([len(out_sources), 1])
            # out_sources=out_sources.type(torch.float)
            ##
            outputs_sources.append(out_sources)
            outputs.append(out_sources)
            
        for party, layer in self.input_party_width.items():
            input_indices = [self.source_indices[source] for source in self.config[party][0]]
            selected_inputs = [outputs_sources[input_index] for input_index in input_indices]
            selected_inputs = torch.cat(selected_inputs, dim=1)
            h = layer(selected_inputs)
            h = self.hidden_width[party](h)
            
            sources = self.config[party][0]
            start_slice=0
            for received_symbols in product(self.all_symbols[party], repeat=len(sources)):
                name=party
                for n_source in range(len(sources)):
                    name=name+"_"+sources[n_source]+received_symbols[n_source]   
                    
                stop_slice = start_slice+self.config[party][1]
                
                out = self.output_width[name](h[:,start_slice:stop_slice])
                outputs.append(out)
                
                start_slice = stop_slice

        joint_prob = torch.cat(outputs, dim=1)
        return joint_prob
    
    def outputsize(self):
        keys= list(self.config.keys())
        n_outputs = 0
        for akey in keys:
            n_outputs+= self.config[akey][1]
        return n_outputs


#############################Generate Data#####################################

def generate_data(config, target_distribution=[]):
    data_type = torch.float32
    
    return torch.ones(1, dtype=data_type), target_distribution

################################Train Model########################

def train_model_and_save(model, n_training_steps=10000, model_path='model/model.pth', threshold=0):
    """main function for training
    
    Parameters:
        - model : an instance of NeuralNetwork
        - n_training_steps=10000 : the number of training steps maximal
        - model_path='model/model.pth' : the path to save the model
        - threshold=0 : distance when stopping the training
    
    Returns:
        model.best_dist : the best distance found during training
    """
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model.to(device)
    
    criterion = CustomLoss(model, dist=model.dist)
    np.set_printoptions(precision=8, suppress=False)
                
    with tqdm(range(n_training_steps)) as t:
        for step in t:
            
            model.train()
    
            data, labels = generate_data(model.config, model.target_distribution)
            data, labels = torch.tensor(data).clone().detach().to(device), torch.tensor(labels).clone().detach().to(device)
            # data, labels = torch.tensor(data).to(device), torch.tensor(labels).to(device)
    
            total_outputs = model(data)
            model.optimizer.zero_grad()
            # loss, predict= criterion(total_outputs, labels, compiled_distribution_function)
            loss, predict= criterion(total_outputs, labels)
            
            loss.backward()
            model.optimizer.step()            

            if loss < model.best_dist:
                model.best_dist = loss
                
                elapsed = t.format_dict['elapsed']
                elapsed_str = t.format_interval(elapsed)
    
                print(f"update model : dist ({model.dist}) =", np.around(float(loss), 5), "time ", elapsed_str)
                save_model_and_optimizer(model, model_path)

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
        'config': model.config,
        'cardinality': model.cardinality,
        'target_distribution': model.target_distribution,
        'width': model.width,
        'depth': model.depth,
        'lr': model.lr,
        'dist': model.dist,
        'scale_blocks_with_n_sources': model.scale_blocks_with_n_sources,
    }, path)


def load_model_and_optimizer(model, path='model/model.pth', print_in_console=True):
    model.to(device)
    checkpoint = torch.load(path, map_location=device)
    
    model.best_dist = checkpoint.get('best_dist', float('inf'))
    model.target_distribution = checkpoint.get('target_distribution')
    model.width = checkpoint.get('width', int(60))
    model.depth = checkpoint.get('depth', int(4))
    model.cardinality = checkpoint.get('cardinality')
    model.config = checkpoint.get('config')
    model.lr = checkpoint.get('lr')
    model.dist = checkpoint.get('dist')
    model.scale_blocks_with_n_sources=checkpoint.get('scale_blocks_with_n_sources')
    
    model.build_model()

    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if print_in_console:
        print("Model and optimizers loaded from :", path)
        print("Best distance {} loaded :".format(model.dist), model.best_dist)



####################################Evaluate######################################


def get_dist_from_model(path):
    """return the distance of a saved model
    inputs : model (configuration of NN), path of saved model
    outputs : distance"""
    model = NeuralNetwork()
    load_model_and_optimizer(model, path, print_in_console=False)
    model.to(device)
        
    checkpoint = torch.load(path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.best_dist = checkpoint.get('best_dist', float('inf'))

    return model.best_dist


def get_total_outputs(model=None, path=None):
    
    if path is not None:
        model = NeuralNetwork()
        load_model_and_optimizer(model, path, print_in_console=False)
        model.to(device)

    elif model is None:
        raise(ValueError("A preloaded model or a path to a model should be given. Usage: get_total_outputs(model=None, path=None) "))

    data, labels = generate_data(model.config)

    total_outputs = model(data)

    return total_outputs        

def evaluate_model(model=None, path=None, dist=None):
    """evaluate a model and return the distance and probability distribution
    One preloaded model or path to a model to load should be given
    inputs: 
        - model=None: a preloaded model
        - path=None: a path to a model to load
        - dist=None: type of distance ("eucl" or "KL"). If None, model.dist is chosen
    outputs:
        - the distance
        - the probability distribution
        """

    if path is not None:
        model = NeuralNetwork()
        load_model_and_optimizer(model, path, print_in_console=False)
        model.to(device)

    elif model is None:
        raise(ValueError("A preloaded model or a path to a model should be given. Usage: evaluate_model(model=None, path=None, dist=None) "))

    if dist is None:
        dist = model.dist

    criterion = CustomLoss(model, dist=dist)
    model.eval() 
    
    data, labels = generate_data(model.config, model.target_distribution)
    data, labels = torch.tensor(data).to(device), torch.tensor(labels).to(device)

    total_outputs = model(data)

    d, predict = criterion(total_outputs, labels)
    
    return d.detach().cpu().numpy(), predict.detach().cpu().numpy()


def plot_distribution(model=None, path=None, dist=None, title_style=1):
    import matplotlib.pyplot as plt
    
    if path is not None:
        model = NeuralNetwork()
        load_model_and_optimizer(model, path, print_in_console=False)
        model.to(device)

    elif model is None:
        raise(ValueError("A preloaded model or a path to a model should be given. Usage: plot_distribution(model=None, path=None, dist=None) "))

    if dist is None:
        dist = model.dist

    criterion = CustomLoss(model, dist)
    model.eval() 

    data, labels = generate_data(model.config, model.target_distribution)
    data, labels = torch.tensor(data).to(device), torch.tensor(labels).to(device)

    total_outputs = model(data)
    d, predict = criterion(total_outputs, labels)
    
    plt.scatter(np.arange(len(predict)), predict.detach().numpy())
    plt.scatter(np.arange(len(model.target_distribution)), model.target_distribution)
    if title_style==0: 
        pass
    elif title_style==1:
        if path is None:
            path=""
        plt.title(path + "  distance ({}) = {}".format(dist, '{0:.6f}'.format(d)))
    elif title_style==2:
        plt.title("distance ({}) = {}".format(dist, '{0:.6f}'.format(d)))
    else:
        print("title_style not defined")
    plt.plot()


## plot local strategies

def process_output(y_pred, output_sizes, model):
    start_idx = 0
    cardinality = model.cardinality
    
    num_sources = len(model.source_indices)
    
    output_sources = y_pred[:, :num_sources*cardinality]
    output_party = y_pred[:, num_sources*cardinality:]
    
    output_sources = torch.reshape(output_sources, [num_sources, cardinality])
    all_party_out = []
    for party, (sources, n_out) in model.config.items():#loop for party to shape the output_party in a convinient way
        n_connected_sources = len(sources)
        end_idx = start_idx + (cardinality**n_connected_sources)*n_out
        this_party_out = output_party[:, start_idx:end_idx]
        dim_array = [cardinality for _ in range(n_connected_sources)]
        dim_array.append(n_out)
        this_party_out = torch.reshape(this_party_out, dim_array)
        all_party_out.append(this_party_out)
        start_idx = end_idx
    return output_sources, all_party_out
    
def get_strategy(model=None, path=None):
    if path is not None:
        model = NeuralNetwork()
        load_model_and_optimizer(model, path, print_in_console=False)
        model.to(device)

    elif model is None:
        raise(ValueError("A preloaded model or a path to a model should be given. Usage: get_strategy(model=None, path=None) "))

    output_sizes = [size for (_, size) in model.config.values()]
    data, labels = generate_data(model.config)

    total_outputs = model(data)
    return process_output(total_outputs, output_sizes, model)

##plot strategies
   
    
def plot_strats_binary(party, source_x, source_y, model=None, path=None):
    """function to plot the strategy as gray scaled plot"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    if path is not None:
        model = NeuralNetwork()
        load_model_and_optimizer(model, path, print_in_console=False)
        model.to(device)

    elif model is None:
        raise(ValueError("A preloaded model or a path to a model should be given."))

    fig=plt.figure(figsize=(3,3), dpi=400)
    ax=fig.add_subplot(1,1,1)
    colors = ['#47a9ff', '#ff8d47', '#5cff47', '#ff474d']
    
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('my_colormap', list(zip([0,1], ["black", "white"])))

    output_sources, all_party_out = get_strategy(model)
    
    sources = list(set(source for person in model.config for source in model.config[person][0]))
    source1 = sources.index(source_x)
    source2 = sources.index(source_y)
    parties=[]
    for aparty in model.config:
        parties.append(aparty)
    index_party = parties.index(party)
    
    id_source_party_1 = model.config[party][0].index(source_x)
    id_source_party_2 = model.config[party][0].index(source_y)
    
    x_start=0
    for id_x, x_width in enumerate(output_sources[source1]):
        y_start=0
        for id_y, y_height in enumerate(output_sources[source2]):
            id_xy = [0 for i in range(len(model.config[party][0]))]
            id_xy[id_source_party_1]=id_x
            id_xy[id_source_party_2]=id_y
            output = float(all_party_out[index_party][id_xy[0]][id_xy[1]][0])
            color = cmap(output)
            ax.add_patch(Rectangle((x_start, y_start), x_width.detach().numpy(), y_height.detach().numpy(), color=color))
            y_start+=y_height.detach().numpy()
        x_start+=x_width.detach().numpy()

    plt.title("Local strategy of " + party)
    plt.xlabel(source_x)
    plt.ylabel(source_y)
    plt.show()

def plot_strats(party, source_x, source_y, model=None, path=None):
    
    if path is not None:
        model = NeuralNetwork()
        load_model_and_optimizer(model, path, print_in_console=False)
        model.to(device)

    elif model is None:
        raise(ValueError("A preloaded model or a path to a model should be given."))
        
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig=plt.figure(figsize=(3,3), dpi=400)
    ax=fig.add_subplot(1,1,1)
    colors = ['#47a9ff', '#ff8d47', '#5cff47', '#ff474d']

    
    output_sources, all_party_out = get_strategy(model, path)
    
    sources = list(set(source for person in model.config for source in model.config[person][0]))
    source1 = sources.index(source_x)
    source2 = sources.index(source_y)
    parties=[]
    for aparty in model.config:
        parties.append(aparty)
    index_party = parties.index(party)
    
    id_source_party_1 = model.config[party][0].index(source_x)
    id_source_party_2 = model.config[party][0].index(source_y)
    
    x_start=0
    for id_x, x_width in enumerate(output_sources[source1]):
        y_start=0
        for id_y, y_height in enumerate(output_sources[source2]):
            id_xy = [0 for i in range(len(model.config[party][0]))]
            id_xy[id_source_party_1]=id_x
            id_xy[id_source_party_2]=id_y
            output = int(all_party_out[index_party][id_xy[0]][id_xy[1]].max(dim=0, keepdim=False)[1])
            ax.add_patch(Rectangle((x_start, y_start), x_width.detach().numpy(), y_height.detach().numpy(), color=colors[output]))
            y_start+=y_height.detach().numpy()
        x_start+=x_width.detach().numpy()

    plt.title("Local strategy of " + party)
    plt.xlabel(source_x)
    plt.ylabel(source_y)
    plt.show()
    
    
    





