from Node import Node
import numpy as np
class Neuron:
    def __init__(self, in_dim, interval):
        self.w = [Node(np.random.uniform(-interval, interval)) for i in range(in_dim)]
        self.b = Node(0)

    def __call__(self, x):
        neuron = Node(0)
        for i in range(len(x)):
            neuron += Node(x[i]) * self.w[i]
        neuron += self.b
        return neuron

    def params(self, ):
        return self.w + [self.b]


class Linear:
    def __init__(self, in_dim, out_dim):
        interval = np.sqrt(6) / np.sqrt(in_dim + out_dim)
        self.neurons = [Neuron(in_dim, interval) for i in range(out_dim)]

    def __repr__(self, ):
        return 'Linear_layer'

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def params(self):
        parameters = []
        for neuron in self.neurons:
            ps = neuron.params()
            parameters.extend(ps)
        return parameters


class Sigmoid:
    def __init__(self, ):
        pass

    def __repr__(self, ):
        return 'Sigmoid_layer'

    def __call__(self, neurons):
        if isinstance(neurons, Node):
            return neurons.sigmoid()
        else:
            for i in range(len(neurons)):
                neurons[i] = neurons[i].sigmoid()
        return neurons

    def params(self, ):
        return []