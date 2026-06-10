import math

def activate_sigmoid(x):
    return 1 / (1 + math.exp(-x))

def propagate_signals(graph, input_neurons, hidden_neurons, output_neurons):
    """Executes a topological sequence feed forward computation loop via graph edges."""
    # Reset tracking activations
    for h in hidden_neurons: h.value = 0.0
    for o in output_neurons: o.value = 0.0

    # Propagate from Input to Hidden
    for input_node in input_neurons:
        for target, weight in graph.adjacency_list.get(input_node, []):
            target.value += input_node.value * weight

    # Apply activation function on Hidden Layer Nodes
    for h in hidden_neurons:
        h.value = activate_sigmoid(h.value)

    # Propagate from Hidden to Output
    for hidden_node in hidden_neurons:
        for target, weight in graph.adjacency_list.get(hidden_node, []):
            target.value += hidden_node.value * weight

    # Apply final activation
    for o in output_neurons:
        o.value = activate_sigmoid(o.value)
