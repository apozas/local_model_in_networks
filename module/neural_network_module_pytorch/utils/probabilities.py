import numpy as np
from itertools import product
import tensorflow as tf


def convert_to_marginal_probabilities(flat_probabilities, variable_sizes):
    """
    Converts a flat list of probabilities into a list of lists of marginal probabilities for each variable,
    where each sublist corresponds to a variable and its marginal probabilities.

    Args:
    flat_probabilities (list): A flat list containing all marginal probabilities for all variables.
    variable_sizes (list): A list where each element specifies the number of values that a variable can take.

    Returns:
    list of lists: A list where each sublist contains the marginal probabilities for one variable.
    """
    marginal_probabilities = []
    start_index = 0

    for size in variable_sizes:
        end_index = start_index + size
        marginal_probabilities.append(flat_probabilities[start_index:end_index])
        start_index = end_index

    return marginal_probabilities





def calculate_joint_probabilities(flat_probabilities, variable_sizes):
    """
    Calculates the joint probability distribution for a set of variables given their marginal probabilities.
    
    Args:
    flat_probabilities (list): A flat list containing all marginal probabilities for all variables.
    variable_sizes (list): A list where each element specifies the number of values that a variable can take.

    Returns:
    numpy.ndarray: An array containing the joint probabilities for all combinations of variable states.
    """
    
    marginal_probabilities = convert_to_marginal_probabilities(flat_probabilities, variable_sizes)
    
    
    index_combinations = product(*(range(len(marginals)) for marginals in marginal_probabilities))
    
    joint_probabilities = []
    
    for indices in index_combinations:
        joint_proba = 1
        for var_index, state_index in enumerate(indices):
            joint_proba *= marginal_probabilities[var_index][state_index]
        joint_probabilities.append(joint_proba)
    
    return np.array(joint_probabilities)




def custom_loss_distribution_np(y_pred, output_sizes):
    start_idx = 0
    prob_tensors = []
    num_segments = len(output_sizes)
    
    for i, size in enumerate(output_sizes):
        end_idx = start_idx + size
        prob_segment = y_pred[:, start_idx:end_idx]

        new_shape = [-1] + [1] * num_segments
        new_shape[i + 1] = size

        reshaped_segment = prob_segment.reshape(new_shape)
        prob_tensors.append(reshaped_segment)
        start_idx = end_idx

    joint_probs = prob_tensors[0]

    for prob in prob_tensors[1:]:
        joint_probs = joint_probs * prob

    joint_probs = np.mean(joint_probs, axis=0)

    joint_probs = joint_probs.flatten()
    return joint_probs





def custom_kl_divergence_np(y_pred, y_true):
    y_pred_clipped = np.clip(y_pred, 1e-10, 1)
    log_y_pred = np.log(y_pred_clipped)
    
    # Calculer la divergence KL en utilisant la formule correcte
    kl_div = np.sum(y_true * (np.log(y_true + 1e-10) - log_y_pred))
    return kl_div






def numpy_list(list_value):
    return np.array(list_value, dtype=np.float32)
