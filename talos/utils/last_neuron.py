def last_neuron(self):

    labels = list(set(self.y.flatten('F')))

    try:
        last_neuron = self.y.shape[1]
        return last_neuron
    except IndexError:
        if len(labels) == 2 and max(labels) == 1:
            last_neuron = 1
        elif len(labels) == 2 and max(labels) > 1:
            last_neuron = 3
        elif len(labels) > 2:
            last_neuron = len(labels)

    return last_neuron
