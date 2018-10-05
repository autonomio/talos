from .reduce_prepare import reduce_prepare
from .reduce_finish import reduce_finish
from .correlation import correlation


def reduce_run(self):

    '''The process run script for reduce
    procedures; takes care of everything
    related with reduction. When new
    reduction methods are added, they need
    to be added as options here.
    '''

    # prepare log for reduction analysis
    self = reduce_prepare(self)

    # run the selected reduction method
    if self.reduction_method == 'correlation':
        self = correlation(self)

    # TODO: the case where reduction_method
    # is not selected or is wrong could be
    # handled better.

    # handle the dropping of permutations
    if self._reduce_keys is None:
        return self
    else:
        return reduce_finish(self)
