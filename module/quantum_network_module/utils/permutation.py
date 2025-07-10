import numpy as np
import qutip as qt
from module.quantum_network_module.utils.array import numpy_array_to_qutip_state


def matrix_representation(dictionary, matrix_id_list):
    """
    Creates a binary matrix that represents the presence of matrix identifiers from matrix_id_list
    in the lists of values for each key in the dictionary.
    
    Parameters:
    - dictionary : A dictionary where each key is associated with a list of matrix identifiers.
    - matrix_id_list : A list of tuples, where each tuple contains a matrix identifier and possibly other data.
    
    Returns:
    - A numpy array (matrix) where each row corresponds to an identifier in matrix_id_list and each column
      corresponds to a key in the dictionary. The entries are 1 if the identifier is present in the list for that key,
      otherwise 0.
    """
    
    # Initialize a zero matrix of appropriate size: rows are matrix IDs, columns are dictionary keys
    matrix = np.zeros((len(matrix_id_list), len(dictionary)), dtype=int)

    # Extract matrix IDs from matrix_id_list
    ids = [id for id, _ in matrix_id_list]
    
    # Extract keys from the dictionary
    keys = list(dictionary.keys())
    
    # Fill the matrix with 1's where the matrix ID is found in the corresponding dictionary list
    for i, matrix_id in enumerate(ids):
        for j, key in enumerate(keys):
            if matrix_id in dictionary[key]:
                matrix[i, j] = 1

    return matrix


def fct_matrice_permutation(matrix):
    """
    Transforms a binary matrix into a permutation list where the order of elements
    is determined by their sequential numbering across the matrix, transposed and filtered.
    
    Parameters:
    - matrix : A numpy array (2D), typically binary (0s and 1s), where 1s will be indexed.
    
    Returns:
    - A list of indices derived from a transposed version of the input matrix where
      each '1' from the original matrix has been replaced by a sequential number
      representing its order. 
      The list will represent the permutation of the source matrix of the network.
    """
    # Copy the matrix to avoid modifying the original matrix
    transformed_matrix = np.copy(matrix)
    counter = 1  # Start counting from 1
    
    # Iterate through each element in the matrix and number the '1' elements sequentially
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[j, i] == 1:
                transformed_matrix[j, i] = counter
                counter += 1
    print(transformed_matrix)
    # Transpose the transformed matrix
    # transposed_matrix = transformed_matrix.T
    transposed_matrix = transformed_matrix

    # Flatten the transposed matrix and filter out zeros
    flattened_list = transposed_matrix.flatten()
    filtered_list = [int(value) - 1 for value in flattened_list if value != 0]

    return filtered_list


def fct_permutation_povm(dict_povm):
    """
    Generates a permutation list for a dictionary where each key represents a measurement and
    the values are the sources received by that measurement. The values (sources) for each 
    measurement (key) are sorted alphabetically, and indices are assigned globally across all 
    measurements based on this sorted order.

    Parameters:
    - dict_povm : A dictionary where each key is a measurement and each value is a list of sources
                  that the measurement receives. This function sorts the sources under each 
                  measurement alphabetically, and indices are assigned globally across all measurements.

    Returns:
    - A list of indices representing the permutation of the sources. This list shows the 
      position of each source within a global ordering of all sources across the dictionary,
      sorted by measurement and then by source within each measurement.

    This approach ensures each source's index is unique and continuous throughout all measurements.
    """
    permutation_list = []
    current_index = 0  # Initialize the current index for global tracking of source positions
    # Iterate through each measurement in the dictionary, sorted alphabetically
    for measurement in sorted(dict_povm.keys()):
        # Retrieve and sort the sources for the current measurement alphabetically
        sorted_sources = sorted(dict_povm[measurement])

        # Map each sorted source to its global index
        local_sorted_indices = {source: idx for idx, source in enumerate(sorted_sources, start=current_index)}

        # Append the local sorted index of each source in the original order of appearance
        for source in dict_povm[measurement]:  # Use the original order here
            permutation_list.append(local_sorted_indices[source])

        # Update the current index to continue numbering for the next measurement
        current_index += len(sorted_sources)

    return permutation_list


############################################################################################################


def sources_permut(permutation_matrix, list_sources_dims):
    """
    Constructs a tensor product of quantum objects from source dimensions and applies a permutation 
    based on the specified permutation matrix.

    Parameters:
    - permutation_matrix : A list or array of integers specifying the new order of subsystems 
                           after the tensor product is formed.
    - list_sources_dims : A list of tuples, each containing the parameters to construct individual
                          quantum objects using the function `numpy_array_to_qutip_state`.

    Returns:
    - A numpy array representing the full matrix of the permuted tensor product of quantum objects.
    """
    # Convert source dimensions to QuTiP quantum objects
    list_sources_Qobj = [numpy_array_to_qutip_state(*source_dims) for source_dims in list_sources_dims]
    
    # Perform tensor product of all quantum objects
    sources_tensor_Qobj = qt.tensor(list_sources_Qobj)

    # Apply the permutation and convert the result to a full matrix
    return sources_tensor_Qobj.permute(permutation_matrix).full()


def povm_permut(permutation_matrix, input_dim_povm, povm_dims):
    """
    Applies a permutation to the dimensions of a POVM represented as a QuTiP quantum object.
    
    Parameters:
    - permutation_matrix : A list or array of integers specifying the new order of subsystems.
    - input_dim_povm : A list of dimensions specifying the tensor product structure of the POVM.
    - povm_dims : The matrix or array that represents the quantum object of the POVM.
    
    Returns:
    - A QuTiP Qobj representing the permuted quantum object of the POVM.
    
    This function rearranges the subsystems of the quantum object according to the specified permutation.
    """
    # Create the quantum object from the given dimensions and the quantum data (povm_dims)
    povm_Qobj = qt.Qobj(povm_dims, dims=input_dim_povm)
    
    # Apply the permutation to the quantum object and return the resulting quantum object
    return povm_Qobj.permute(permutation_matrix)



def generate_tensor_product_dims(source_dims):
    """
    Generates tensor product dimensions for composite quantum systems in QuTiP from a list of subsystem dimensions.

    Parameters:
    - source_dims : A list of tuples, where each tuple contains an identifier and the dimensions of a subsystem.
                    The identifiers are ignored, and the dimensions are used to construct the tensor product.

    Returns:
    - A list with two identical lists: one for ket dimensions and one for bra dimensions, formatted to QuTiP's requirements.
    """
    # Initialize dimension lists for kets and bras
    ket_dims, bra_dims = [], []

    # Extract dimensions from tuples, ignoring identifiers, and append appropriately
    for _, dim in source_dims:
        if not isinstance(dim, (list, tuple)):
            dim = [dim]  # Ensure the dimension is iterable
        ket_dims.extend(dim)
        bra_dims.extend(dim)

    return [ket_dims, bra_dims]
