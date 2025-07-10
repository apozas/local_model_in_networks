import numpy as np
from itertools import product

def possible_combinations_povm(povm_list):
    """
    Returns all possible combinations of POVM measurement indices.
    
    Parameters:
    - povm_list : A list of tuples, each containing a measurement name and a list of POVMs for that measurement.

    Returns:
    - A list of tuples, each containing indices representing the combinations of outcomes for all POVMs across all measurements.
    """

    return list(product(*[range(len(povm[0])) for povm in povm_list]))


def possible_combinations_list(value_list):
    """
    Returns all possible combinations of indices for each value in the list.
    
    Parameters:
    - value_list : A list of integers, each integer represents the number of possible outcomes.

    Returns:
    - A list of tuples, each tuple containing a combination of indices based on the number of possible outcomes for each element in the value_list.
    """

    return list(product(*[range(value) for value in value_list]))
