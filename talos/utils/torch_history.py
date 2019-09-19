class TorchHistory:

    '''This is a helper for replicating the history object
    behavior of Keras to make Talos Scan() API consistent between
    the two backends.'''

    def __init__(self):

        self.history = {}

    def init_history(self):
        self.history = {}

    def append_history(self, history_data, label):
        if label not in self.history.keys():
            self.history[label] = []
        self.history[label].append(history_data)

    def append_loss(self, _loss):
        self.append_history(_loss, 'loss')

    def append_metric(self, _metric):
        self.append_history(_metric, 'metric')

    def append_val_loss(self, _loss):
        self.append_history(_loss, 'val_loss')

    def append_val_metric(self, _loss):
        self.append_history(_loss, 'val_metric')
