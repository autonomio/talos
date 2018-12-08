from numpy import arange, unique, array, column_stack
from itertools import product

from ..reducers.sample_reducer import sample_reducer


class ParamGrid:

    '''Suite for handling parameters internally within Talos

    Takes as input the parameter dictionary from the user, and
    returns a class object which can then be used to pick parameters
    for each round together with other parameter related operations.

    '''

    def __init__(self, main_self):

        self.main_self = main_self

        # convert the input to useful format
        self._p = self._param_input_conversion()

        # build the parameter permutation grid
        self.param_grid = self._param_grid()
        
        # reduce according to downsample
        if self.main_self.grid_downsample is not None:
            self.param_grid = sample_reducer(self)

        # create a index for logging purpose
        self.param_log = list(range(len(self.param_grid)))

        # add the log index to param grid
        self.param_grid = column_stack((self.param_grid, self.param_log))




    def _param_grid(self):

        '''CREATE THE PARAMETER PERMUTATIONS

        This is done once before starting the experiment.
        Takes in the parameter dictionary, and returns
        every possible permutation in an array.
        '''

        ls = [list(self._p[key]) for key in self._p.keys()]
        _param_grid_out = array(list(product(*ls)), dtype='object')

        return _param_grid_out

    def _param_input_conversion(self):

        '''DETECT PARAM FORMAT

        Checks of the hyperparameter input format is list
        or tupple in the params dictionary and expands accordingly.

        '''

        out = {}

        for param in self.main_self.params.keys():

            # for range/step style input
            if isinstance(self.main_self.params[param], tuple):
                out[param] = self._param_range(self.main_self.params[param][0],
                                               self.main_self.params[param][1],
                                               self.main_self.params[param][2])
            # all other input styles
            else:
                out[param] = self.main_self.params[param]

        return out

    def _param_range(self, start, end, n):

        '''PARAMETER RANGE

        Deals with the format where a start, end
        and steps values are given for a parameter
        in a tuple format.

        This is called internally from param_format()
        '''

        try:
            out = arange(start, end, (end - start) / n, dtype=float)
        # this is for python2
        except ZeroDivisionError:
            out = arange(start, end, (end - start) / float(n), dtype=float)

        if type(start) == int and type(end) == int:
            out = out.astype(int)
            out = unique(out)

        return out
