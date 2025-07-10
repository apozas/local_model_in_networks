from module.quantum_network_module.utils.combinaison import possible_combinations_povm, possible_combinations_list
from module.quantum_network_module.utils.permutation import fct_matrice_permutation, matrix_representation, sources_permut, povm_permut, fct_permutation_povm, generate_tensor_product_dims
from module.quantum_network_module.utils.display import display_network
from module.quantum_network_module.utils.array import assure_numpy_arrays
import numpy as np
# import matplotlib
# # Force matplotlib to use a specific GUI backend, e.g., 'TkAgg'
# matplotlib.use('TkAgg')


def calculer_proba(source_tensor, list_povm, combinaisons_input, combinaisons_output, tuple_info_povm=None):
    """
    Calculates the probability for a given configuration of a quantum network.

    Parameters:
    - source_tensor : The quantum state matrix representing the source system.
    - list_povm : A list of POVM, where each sublist contains matrices representing the possible outcomes of a POVM.
    - combinaisons_input : A list of indices specifying which set of POVMs to apply.
    - combinaisons_output : A list of indices specifying which outcome from each selected POVM set to consider.
    - tuple_info_povm : Optional tuple containing additional parameters for permution of the resulting POVM tensor.

    Returns:
    - The calculated probability as a float, representing the trace of the product of the resulting POVM tensor
      and the source tensor, rounded to 10 decimal places.

    The function uses tensor products to combine POVM (applies an optional permutation if provided) with the source tensor, 
    and then calculates the probability by taking the trace of the resulting tensor product.
    """
    # Initialize the POVM tensor with the first selected outcome
    povm_tensor = list_povm[0][combinaisons_input[0]][combinaisons_output[0]]

    # Use tensor products to build the full POVM tensor from the selected outcomes
    for i in range(1, len(combinaisons_output)):
        povm_tensor = np.kron(povm_tensor, list_povm[i][combinaisons_input[i]][combinaisons_output[i]])

    # If a permutation is specified, apply it to the POVM tensor
    if tuple_info_povm is not None:
        povm_tensor = povm_permut(*tuple_info_povm, povm_tensor)

    # Calculate the probability by taking the trace of the product of the POVM tensor and the source tensor
    return np.float32(np.round(np.trace(np.dot(povm_tensor, source_tensor)), 10))




def is_condition_satisfied(source_tensor, list_povm, combinaisons_output, tuple_info_povm=None):
    """
    Determines if the conditions for conditional probabilities are met and calculates those probabilities.

    Parameters:
    - source_tensor: The quantum state or tensor representing the source systems.
    - list_povm: A list of POVMs applied to the quantum system.
    - combinaisons_output: The specific output combination for which the probability is calculated.
    - tuple_info_povm: Optional additional data for POVMs, such as permutations or specific dimensions.

    Returns:
    - A list of probabilities that reflect either the direct or conditional probabilities depending on
      the number of elements in list_povm.
    """
    # Determine the number of outcomes for each POVM
    combinaison_input = [len(povm) for povm in list_povm]
    
    # Generate all possible combinations of outcomes for the given POVMs
    combinaisons_input = possible_combinations_list(combinaison_input)
    
    # Single combination scenario: directly calculate and print the probability
    if len(combinaisons_input) == 1:
        prob_result = calculer_proba(source_tensor, list_povm, combinaisons_input[0], combinaisons_output, tuple_info_povm)
        #print(f"p{combinaisons_output} = {prob_result}") ## print the probability
        return [prob_result]
    else:
        # Multiple combinations scenario: calculate conditional probabilities
        result_proba_list = []
        for combin_input in combinaisons_input:
            prob_result = calculer_proba(source_tensor, list_povm, combin_input, combinaisons_output, tuple_info_povm)
            #print(f"p({', '.join(map(str, combinaisons_output))} | {', '.join(map(str, combin_input))}) = {prob_result}") ## print the probability
            result_proba_list.append(prob_result)
        
        return result_proba_list
       

def noise(rho, noise):
    matrice_I = np.eye(4)
    norm = np.linalg.norm(matrice_I)
    matriceI_norm = matrice_I / norm
    rho = np.array(rho)  # Convert input to a NumPy array if it isn't one already
    return rho * noise + matriceI_norm * (1-noise)



def quantum_network_fct(graph, source_dims_list, povm_list, povm_dims_list=None, permutation=None):
    """
    Calculates the probability distribution for a given quantum network.

    Parameters:
    - graph : Dictionary representing connections between sources and measurements.
    - source_dims_list : A list of tuples indicating the sources and their dimensions.
    - povm_list : A list of tuples with each tuple containing an identifier and a matrix representing a POVM.
    - povm_dims_list : Optional list of dimensions for the POVMs; used for advanced configurations.
    - permutation : Optional list of [permutation_povm, permutation_sources], to specify a specific permutation

    Returns:
    - A list of probability distributions across the network based on the provided POVMs and source configuration.
    """
    # Optional: Visual display of the network (uncomment if visualization is needed)
    # display_network(graph)

    # Generate a permutation matrix based on the network graph and POVMs
    if permutation is None:
        permutation_matrix = fct_matrice_permutation(matrix_representation(graph, povm_list))
    else:
        permutation_matrix = permutation[1]
    print("Source permutation matrix:", permutation_matrix)

    # Assure all source dimensions are numpy arrays
    source_dims_list = assure_numpy_arrays(source_dims_list, "source")

    # Permute source tensors according to the permutation matrix
    source_tensor_permut = sources_permut(permutation_matrix, source_dims_list)

    # Extract matrices from povm_list
    povm_matrices = [matrice for _, matrice in povm_list]

    # Generate all possible combinations of outcomes for the given POVMs
    combinations = possible_combinations_povm(povm_matrices)

    probability_distribution = []

    if povm_dims_list:
        # If POVM dimensions are provided, generate tensor product dimensions for them
        input_dim_povm = generate_tensor_product_dims(source_dims_list)

        # Generate a permutation for POVMs
        if permutation is None:
            permutation_povm = fct_permutation_povm(povm_dims_list)
        else:
            permutation_povm = permutation[0]
        tuple_info_povm = (permutation_povm, input_dim_povm)

        print("POVM permutation matrix:", permutation_povm)

        # Calculate probabilities for each combination considering POVM permutations
        for combination in combinations:
            probability_distribution += is_condition_satisfied(source_tensor_permut, povm_matrices, combination, tuple_info_povm)
    else:
        print("POVM permutation matrix: default (alphabetical order)")
        # Calculate probabilities for each combination without specific POVM permutations
        for combination in combinations:
            probability_distribution += is_condition_satisfied(source_tensor_permut, povm_matrices, combination)

    return probability_distribution

