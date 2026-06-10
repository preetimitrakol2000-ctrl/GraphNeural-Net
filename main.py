from node import NeuronNode
from graph import NetworkGraph
from forward_pass import propagate_signals

if __name__ == "__main__":
    print("🕸️  Assembling Topological Directed Graph GraphNeural-Net...")
    
    # Initialize explicit network node layout blueprints
    i1, i2 = NeuronNode("Input_1"), NeuronNode("Input_2")
    h1, h2 = NeuronNode("Hidden_1"), NeuronNode("Hidden_2")
    o1 = NeuronNode("Output_1")

    # Seed static parameters
    i1.value = 0.8
    i2.value = 0.4

    nn_graph = NetworkGraph()
    nn_graph.add_neuron(i1); nn_graph.add_neuron(i2)
    nn_graph.add_neuron(h1); nn_graph.add_neuron(h2)

    # Wire up layer synapses with static weight variables
    nn_graph.add_synapse(i1, h1, 0.5)
    nn_graph.add_synapse(i1, h2, -0.2)
    nn_graph.add_synapse(i2, h1, 0.7)
    nn_graph.add_synapse(i2, h2, 0.1)
    nn_graph.add_synapse(h1, o1, 0.9)
    nn_graph.add_synapse(h2, o1, -0.4)

    propagate_signals(nn_graph, [i1, i2], [h1, h2], [o1])

    print(f"📥 Feeder Signals Layer Arrays: [{i1.value}, {i2.value}]")
    print(f"🔮 Final Forward Path Graph Output Score: {o1.value:.4f}")
