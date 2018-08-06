from .reduce_prepare import reduce_prepare
from .reduce_finish import reduce_finish
from .correlation import correlation


def reduce_run(self):

    self = reduce_prepare(self)

    if self.reduction_method == 'correlation':
        self = correlation(self)

    if self._reduce_keys is None:
        return self
    else:
        return reduce_finish(self)
