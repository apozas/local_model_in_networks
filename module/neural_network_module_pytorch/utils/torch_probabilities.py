import torch.nn as nn
import torch


def custom_loss_distribution(y_pred, output_sizes):
    start_idx = 0
    prob_tensors = []
    num_segments = len(output_sizes)
    

    for i, size in enumerate(output_sizes):
        end_idx = start_idx + size
        prob_segment = y_pred[:, start_idx:end_idx]

        new_shape = [-1] + [1] * num_segments
        new_shape[i + 1] = size
        reshaped_segment = prob_segment.view(new_shape)
        prob_tensors.append(reshaped_segment)
        start_idx = end_idx

    joint_probs = prob_tensors[0]
    for prob in prob_tensors[1:]:
        joint_probs = joint_probs * prob


    joint_probs = torch.mean(joint_probs, dim=0)
    joint_probs = joint_probs.flatten()
    return joint_probs


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
    def __init__(self, config, dist='kl'):
        super(CustomLoss, self).__init__()
        self.config = config
        self.output_sizes = [size for (_, size) in config.values()]
        self.dist = dist

    def forward(self, y_pred, y_true, compiled_distribution_function=False):
        if compiled_distribution_function is False:
            probs = custom_loss_distribution(y_pred, self.output_sizes)
        else:
            probs = compiled_distribution_function(y_pred, self.output_sizes)
        
        if self.dist == "kl":
            d = custom_kl_divergence(probs, y_true[0, :])
        elif self.dist == "eucl":
            d = torch.norm(y_true[0, :] - probs)
        else:
            raise ValueError("Distance not defined: should be one of : kl, eucl")
        
        return d, probs



def torch_list(list_value):
    return torch.tensor(list_value, dtype=torch.float64)
