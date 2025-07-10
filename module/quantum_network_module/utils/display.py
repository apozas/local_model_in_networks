import networkx as nx
import matplotlib.pyplot as plt

def display_network(dictionary):
    """
    Visualizes a network from a dictionary, displaying sources at the top and measurements at the bottom.
    
    Parameters:
    - dictionary : A dictionary where keys represent source nodes and values are lists of measurement nodes.

    This function constructs a directed graph where each source node is connected to the corresponding
    measurement nodes. It visually separates source nodes (upper layer) from measurement nodes (lower layer).
    """
    G = nx.DiGraph()
    
    # Add nodes for the sources
    for source in dictionary.keys():
        G.add_node(source, layer=0)
    
    # Add nodes for the measurements and create edges from sources to measurements
    for source, values in dictionary.items():
        for value in values:
            measurement = value  # Directly use the identifiers from the dictionary
            if not G.has_node(measurement):
                G.add_node(measurement, layer=1)
            G.add_edge(source, measurement)
    
    # Positioning the nodes
    pos = {}
    sources = [node for node, data in G.nodes(data=True) if data['layer'] == 0]
    measurements = [node for node, data in G.nodes(data=True) if data['layer'] == 1]
    width = max(len(sources), len(measurements))
    
    # Position sources at the top, centered relative to measurements
    for i, source in enumerate(sources):
        pos[source] = (i - len(sources)/2 + width/2, 1)
    
    # Position measurements at the bottom
    for i, measurement in enumerate(measurements):
        pos[measurement] = (i - len(measurements)/2 + width/2, 0)
    
    # Draw the graph
    plt.figure(figsize=(15, 8))
    nx.draw(G, pos, with_labels=True, arrows=True, node_size=2000,
            node_color=["tomato" if data["layer"] == 0 else "lightgreen" for node, data in G.nodes(data=True)],
            font_size=12, font_weight="bold", edge_color="gray")
    
    plt.title("Network Visualization")
    plt.axis('off')
    plt.show()
    
    
def pretty_print(result, precision=4):
    """
    Nicely prints a list of complex numbers, grouping them by a specified number per line.
    
    Parameters:
    - result : A list of complex numbers.
    - precision : The number of complex numbers to print per line, defaulting to 4.

    This function formats each complex number to display its real and imaginary parts with six decimal places.
    """
    for i in range(0, len(result), precision):
        line = ' '.join(f"{complex_num.real:.6f}" for complex_num in result[i:i+precision])
        print(line)


