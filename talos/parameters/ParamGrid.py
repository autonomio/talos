import numpy as np

from ..reducers.sample_reducer import sample_reducer
from ..reducers.permutation_filter import permutation_filter


class ParamGrid:

    '''Suite for handling parameters internally within Talos

    Takes as input the parameter dictionary from the user, and
    returns a class object which can then be used to pick parameters
    for each round together with other parameter related operations.

    '''

    def __init__(self, main_self):

        self.main_self = main_self

        # creates a reference dictionary for column number to label
        self.param_reference = {}
        for i, col in enumerate(self.main_self.params.keys()):
            self.param_reference[col] = i

        # convert the input to useful format
        self._p = self._param_input_conversion()

        # create a list of lists, each list being a parameter sequence
        ls = [list(self._p[key]) for key in self._p.keys()]

        # get the number of total dimensions / permutations
        virtual_grid_size = 1
        for l in ls:
            virtual_grid_size *= len(l)
        final_grid_size = virtual_grid_size

        # calculate the size of the downsample
        if self.main_self.grid_downsample is not None:
            final_grid_size = int(virtual_grid_size * self.main_self.grid_downsample)

        # take round_limit into account
        if self.main_self.round_limit is not None:
            final_grid_size = min(final_grid_size, self.main_self.round_limit)

        # create the params grid
        self.param_grid = self._create_param_grid(ls,
                                                  final_grid_size,
                                                  virtual_grid_size)

        # handle the case where permutation filter is provided
        if self.main_self.permutation_filter is not None:
            self = permutation_filter(self,
                                      ls,
                                      final_grid_size,
                                      virtual_grid_size)

        # initialize with random shuffle if needed
        if self.main_self.shuffle:
            np.random.shuffle(self.param_grid)

        # create a index for logging purpose
        self.param_log = list(range(len(self.param_grid)))

        # add the log index to param grid
        self.param_grid = np.column_stack((self.param_grid, self.param_log))

    def _create_param_grid(self, ls, final_grid_size, virtual_grid_size):

        # select permutations according to downsample
        if final_grid_size < virtual_grid_size:
            out = sample_reducer(self, final_grid_size, virtual_grid_size)
        else:
            out = range(0, final_grid_size)

        # build the parameter permutation grid
        param_grid = self._create_param_permutations(ls, out)

        return param_grid

    def _create_param_permutations(self, ls, permutation_index):

        '''Expand params dictionary to permutations

        Takes the input params dictionary and expands it to
        actual parameter permutations for the experiment.
        '''

        final_grid = []
        for i in permutation_index:
            p = []
            for l in reversed(ls):
                i, s = divmod(int(i), len(l))
                p.insert(0, l[s])
            final_grid.append(tuple(p))

        _param_grid_out = np.array(final_grid, dtype='object')

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

        '''Deal with ranged inputs in params dictionary

        A helper function to handle the cases where params
        dictionary input is in the format (start, end, steps)
        and is called internally through ParamGrid().
        '''

        try:
            out = np.arange(start, end, (end - start) / n, dtype=float)
        # this is for python2
        except ZeroDivisionError:
            out = np.arange(start, end, (end - start) / float(n), dtype=float)

        if type(start) == int and type(end) == int:
            out = out.astype(int)
            out = np.unique(out)

        return out
