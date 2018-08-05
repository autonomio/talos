from .ReductionTable import ReductionTable
from .Reducers import Reducers
from .reduce_drop import reduction_drop


def reduce_run(self):

    '''Takes in the Scan object, and returns a modified version
    of the self.param_log.'''

    self._filaname = self.experiment_name + '.csv'

    # create the table for reduction
    out = ReductionTable(self._filaname,
                         self.reduction_metric,
                         self.reduction_window,
                         self.reduction_threshold)

    # create the reducer object
    out = Reducers(out)

    # apply the reduction
    if self.reduction_method == 'correlation':
        self.out = out.correlation()

    if self.out is None:
        return self.param_log
    else:
        return reduction_drop(self)
