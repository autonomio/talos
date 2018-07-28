from numpy import arange, unique, array
from itertools import product

from ..reducers.sample_reducer import sample_reducer


class ParamGrid:

    '''Suite for handling parameters internally within Talos

    Takes as input the parameter dictionary from the user, and
    returns a class object which can then be used to pick parameters
    for each round together with other paramater related operations.

    '''

    def __init__(self, params, search_method, grid_downsample):

        # load the variables
        self.param_dict = params
        self.search_method = search_method
        self.grid_downsample = grid_downsample

        # convert the input to useful format
        self.p = self._param_input_conversion()

        # build the parameter permutation grid
        self.param_grid = self._param_grid()

        # reduce according to downsample
        if self.grid_downsample is not None:
            self.param_grid = sample_reducer(self)

        # create a index for logging purpose
        self.param_log = list(range(len(self.param_grid)))


    def _param_grid(self):

        '''CREATE THE PARAMETER PERMUTATIONS

        This is done once before starting the experiment.
        Takes in the parameter dictionary, and returns
        every possible permutation in an array.
        '''

        ls = [list(self.p[key]) for key in self.p.keys()]
        _param_grid_out = array(list(product(*ls)))

        return _param_grid_out


    def _param_input_conversion(self):

        '''DETECT PARAM FORMAT'''

        out = {}

        for param in self.param_dict.keys():

            # for range/step style input
            if isinstance(self.param_dict[param], tuple):
                out[param] = self.param_range(self.param_dict[param][0],
                                              self.param_dict[param][1],
                                              self.param_dict[param][2])
            # all other input styles
            else:
                out[param] = self.param_dict[param]

        return out


    def param_range(self, start, end, n):

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
