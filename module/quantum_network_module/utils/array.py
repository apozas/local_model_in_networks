import numpy as np
import qutip as qt

def numpy_array_to_qutip_state_4x4(numpy_array):
    """
    Converts a 4x4 numpy array into a QuTiP Qobj representing a quantum state
    of two qubits, with the correct structure of the composite Hilbert space.
    
    Parameters:
    - numpy_array : A 4x4 numpy array representing the density matrix of a
                    two-qubit state.
                    
    Returns:
    - A QuTiP Qobj representing the quantum state of the two qubits, with
      the correct structure of the composite Hilbert space.
    """
    # Verify that the array is the correct size
    if numpy_array.shape != (4, 4):
        raise ValueError("Array must be of size 4x4.")
    
    # Convert the numpy array into a QuTiP Qobj
    qutip_state = qt.Qobj(numpy_array, dims=[[2, 2], [2, 2]])

    
    return qutip_state   

def numpy_array_to_qutip_state(numpy_array, dimensions):
    """
    Converts a numpy array into a QuTiP Qobj representing a quantum state or operator
    in a composite Hilbert space, with a specific dimension structure.
    
    Parameters:
    - numpy_array : A numpy array representing the density matrix or operator of a quantum
                    state or system.
    - dimensions : A list describing the number of states for each quantum subsystem in
                   the composite Hilbert space (e.g., [2, 2] for two qubits, [3, 2, 3] for a
                   qutrit, a qubit, and another qutrit).
                   
    Returns:
    - A QuTiP Qobj representing the quantum state or operator, with the correct structure
      of the composite Hilbert space.
    """
    # Calculate the product of the dimensions to verify the matrix size
    prod_dims = np.prod(dimensions)
    if numpy_array.size != prod_dims**2:
        raise ValueError(f"The array size does not match the provided dimensions. Expected size: {prod_dims}x{prod_dims}.")
    
    input_dim = []
    for dim in dimensions:
        input_dim.append([dim, dim])

    # Convert the numpy array into a QuTiP Qobj with the correct dimensions
    qutip_obj = qt.Qobj(numpy_array, dims=input_dim)
    
    return qutip_obj


def assure_numpy_arrays(liste, nom_liste):
    """
    Ensures that all elements in a list are numpy arrays.
    Converts non-numpy elements into numpy arrays.

    Parameters:
    - liste : The list of elements to check and convert if necessary.
    - nom_liste : The name of the list which can influence how elements are processed.

    Returns:
    - A list where all elements are numpy arrays
    """
    
    if nom_liste == "source":
        liste_modifiee = [(np.array(el[0]), el[1]) if not isinstance(el[0], np.ndarray) else el for el in liste]
        return liste_modifiee
    else:
        return [np.array(el) if not isinstance(el, np.ndarray) else el for el in liste]