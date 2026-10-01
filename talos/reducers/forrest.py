from .trees import _tree_reduce


def forrest(self):
    return _tree_reduce(self, forest=True)
