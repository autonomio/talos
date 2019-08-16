class TorchHistory:

    def __init__(self):

        self.history = {'loss': []}

    def append(self, tensor):

        '''Takes in a tensor for loss or other criterion
        from PyTorch and stores a python scalar in the history
        object. This is for unifying the `Scan()` API between
        Keras and PyTorch'''

        self.history['loss'].append(tensor.data.item())
