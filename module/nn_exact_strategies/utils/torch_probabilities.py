import torch.nn as nn
import torch
import math
from itertools import product
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def custom_loss_distribution(y_pred, output_sizes, model):
    start_idx = 0
    prob_tensors = []
    cardinality = model.cardinality
    num_party = len(output_sizes)
    
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
    
    prob_out = torch.zeros(output_sizes).to(device)
    all_outputs = [range(output_sizes[i]) for i in range(len(output_sizes))]
    
    parties = [party for party, (sources, n_out) in model.config.items()]
    
    # for i in range(math.prod(output_sizes)):
    for received_symbols in product(range(cardinality), repeat=num_sources):
        weight_sources=1
        for id_source in range(num_sources):
            weight_sources *= output_sources[id_source][received_symbols[id_source]]
        
        for outputs in product(*all_outputs):
            weight_parties=1
            for id_party, party in enumerate(parties):
                received_symbols_party = tuple(received_symbols[model.source_indices[model.config[party][0][i]]] for i in range(len(model.config[party][0])))
                weight_parties *= all_party_out[id_party][received_symbols_party][outputs[id_party]]
      
            prob_out[outputs] += weight_sources*weight_parties

    prob_out = prob_out.flatten()
    return prob_out

# def custom_loss_distribution(y_pred, output_sizes, model):
#     start_idx = 0
#     prob_tensors = []
#     cardinality = model.cardinality
#     num_party = len(output_sizes)
    
#     num_sources = len(model.source_indices)
    
#     output_sources = y_pred[:, :num_sources*cardinality]
#     output_party = y_pred[:, num_sources*cardinality:]
    
#     output_sources = torch.reshape(output_sources, [num_sources, cardinality])
#     all_party_out = []
#     for party, (sources, n_out) in model.config.items():#loop for party to shape the output_party in a convinient way
#         n_connected_sources = len(sources)
#         end_idx = start_idx + (cardinality**n_connected_sources)*n_out
#         this_party_out = output_party[:, start_idx:end_idx]
#         dim_array = [cardinality for _ in range(n_connected_sources)]
#         dim_array.append(n_out)
#         this_party_out = torch.reshape(this_party_out, dim_array)
#         all_party_out.append(this_party_out)
#         start_idx = end_idx
    
#     prob_out = torch.zeros(output_sizes).to(device)
#     all_outputs = [range(output_sizes[i]) for i in range(len(output_sizes))]
    
#     parties = [party for party, (sources, n_out) in model.config.items()]
#     for outputs in product(*all_outputs):
#     # for i in range(math.prod(output_sizes)):
#         for received_symbols in product(range(cardinality), repeat=num_sources):
#             weight_sources=1
#             for id_source in range(num_sources):
#                 weight_sources *= output_sources[id_source][received_symbols[id_source]]
#             weight_parties=1
#             for id_party, party in enumerate(parties):
#                 received_symbols_party = tuple(received_symbols[model.source_indices[model.config[party][0][i]]] for i in range(len(model.config[party][0])))
#                 weight_parties *= all_party_out[id_party][received_symbols_party][outputs[id_party]]
#             prob_out[outputs] += weight_sources * weight_parties

    
#     prob_out = prob_out.flatten()
#     return prob_out

def custom_kl_divergence(y_pred, y_true):
    y_pred_clipped = torch.clamp(y_pred, 1e-10, 1)
    log_y_pred = torch.log(y_pred_clipped)


    if y_true.dim() == 1:
        kl_div = torch.sum(y_true * (torch.log(y_true + 1e-10) - log_y_pred))
    else:
        kl_div = torch.sum(y_true * (torch.log(y_true + 1e-10) - log_y_pred), dim=1)
        kl_div = torch.mean(kl_div)

    return kl_div

def K_S_criterion(y_pred, y_true):
    """max of the difference between both distribution, from Kolmogorov-Smirnov statistic"""
    D=torch.abs(torch.max(y_pred-y_true))
    return D


class CustomLoss(nn.Module):
    def __init__(self, model, dist='kl'):
        super(CustomLoss, self).__init__()
        self.config = model.config
        self.output_sizes = [size for (_, size) in model.config.values()]
        self.dist = dist
        self.model = model

    def forward(self, y_pred, y_true, compiled_distribution_function=False):
        if compiled_distribution_function is False:
            probs = custom_loss_distribution(y_pred, self.output_sizes, self.model)
        else:
            probs = compiled_distribution_function(y_pred, self.output_sizes, self.model)
        
        if self.dist == "kl":
            d = custom_kl_divergence(probs, y_true)
        elif self.dist == "eucl":
            d = torch.norm(y_true - probs)
        else:
            raise ValueError("Distance not defined: should be one of : kl, eucl")
        
        return d, probs



def torch_list(list_value):
    return torch.tensor(list_value, dtype=torch.float64)
