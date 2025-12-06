import matplotlib.pyplot as plt
from itertools import product

import numpy as np


def plot_comparison_scatter(target_distribution, predicted_distribution):
    """
    Plot a scatter diagram comparing two probability distributions, with an added line of perfect correlation to visualize how closely the predicted distribution matches the target distribution.

    Args:
    target_distribution (list or numpy array): The distribution of probabilities that serves as the benchmark or target.
    predicted_distribution (list or numpy array): The distribution of probabilities that has been predicted by a model or some other means.

    Output:
    A scatter plot displaying the target distribution on the x-axis and the predicted distribution on the y-axis. Points represent individual probabilities from both distributions plotted against each other. A dashed line on the plot shows where the points would lie if the predicted probabilities perfectly matched the target probabilities. The plot includes a legend and is displayed on a grid for better readability.
    """
    target_distribution = np.array(target_distribution)
    predicted_distribution = np.array(predicted_distribution)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(target_distribution, predicted_distribution, alpha=0.6, color='blue', label='Data Points')
    
    max_val = max(target_distribution.max(), predicted_distribution.max())
    min_val = min(target_distribution.min(), predicted_distribution.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Fit')
    
    plt.title('Scatter Plot of Target vs Predicted Probabilities')
    plt.xlabel('Target Distribution')
    plt.ylabel('Predicted Distribution')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_scatter_distributions(target_distribution, predicted_distribution):
    plt.figure(figsize=(8,6))
    plt.ylim(0,1.1*max([max(target_distribution), max(predicted_distribution)]))
    plt.scatter(range(len(target_distribution)), target_distribution, label="target_distribution")
    plt.scatter(range(len(target_distribution)), predicted_distribution, label="generated distribution")
    plt.legend()
    plt.show

def plot_comparison_density(target_distribution, predicted_distribution):
    """
    Plot density graphs for two distributions: a target and a predicted distribution. This visualization helps to compare the overall shape and spread of both distributions to assess how similar they are.

    Args:
    target_distribution (list or numpy array): The distribution of probabilities that serves as the benchmark or target. This distribution will be visualized in blue.
    predicted_distribution (list or numpy array): The distribution of probabilities predicted by a model or calculated through other means. This distribution will be visualized in orange.

    Output:
    A density plot where the x-axis represents the range of probability values and the y-axis shows the density of these probabilities. The graph will display two overlaid density curves, one for the target distribution and another for the predicted distribution, making it easy to visually compare their shapes and peaks. The plot includes a legend and titles for clarity.
    """
    import seaborn as sns
    plt.figure(figsize=(10, 7))
    sns.kdeplot(target_distribution, label='Target Distribution', color='blue', fill=True)
    sns.kdeplot(predicted_distribution, label='Predicted Distribution', color='orange', fill=True)
    plt.title('Density Plot of Target vs Predicted Distributions')
    plt.xlabel('Probability')
    plt.legend()
    plt.show()
    
    
def generate_labels(value_ranges):
    """
    Generate state labels for given number of values per dimension.
    Args:
    value_ranges (list): A list of integers where each integer specifies the number of values in that dimension.
    """
    ranges = [range(values) for values in value_ranges]
    
    combinations = list(product(*ranges))
    
    labels = [f"p({','.join(map(str, combo))})" for combo in combinations]
    
    return labels
    
    
def plot_distributions(target_distribution, joint_probabilities, list_value):
    """
    Plot the target and predicted distributions using scatter plots with dynamic x-axis labels based on list values.
    Args:
    target_distribution (list or numpy array): The target distribution.
    joint_probabilities (list or numpy array): The joint probabilities for the predicted distribution.
    list_value (list): List of categories or labels for the x-axis.
    """
    
    labels = generate_labels(list_value)

    fig, ax = plt.subplots()
    offset = 0.17
    ind = np.arange(len(target_distribution))

    ax.scatter(ind - offset, target_distribution, label='Distribution Cible', color='blue', marker='o')

    ax.scatter(ind + offset, joint_probabilities, label='Distribution Prédite', color='orange', marker='o')

    ax.set_title('Comparaison des distributions cible et prédite')
    ax.set_xticks(ind)
    if len(labels) < 10:
        ax.set_xticklabels(labels, rotation=45)
    ax.legend()

    plt.show()
    
