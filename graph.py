class NetworkGraph:
    def __init__(self):
        # Adjacency List layout mapping: { source_neuron: [(destination_neuron, weight_value)] }
        self.adjacency_list = {}

    def add_neuron(self, neuron):
        if neuron not in self.adjacency_list:
            self.adjacency_list[neuron] = []

    def add_synapse(self, from_neuron, to_neuron, weight):
        self.adjacency_list[from_neuron].append((to_neuron, weight))
