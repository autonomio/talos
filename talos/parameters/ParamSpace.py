import inspect

import numpy as np
import itertools as it
from datetime import datetime


class ParamSpace:

    def __init__(self,
                 params,
                 param_keys,
                 random_method='uniform_mersenne',
                 fraction_limit=None,
                 round_limit=None,
                 time_limit=None,
                 boolean_limit=None):

        # set all the arguments
        self.params = params
        self.param_keys = param_keys
        self.fraction_limit = fraction_limit
        self.round_limit = round_limit
        self.time_limit = time_limit
        self.boolean_limit = boolean_limit
        self.random_method = random_method

        # set a counter
        self.round_counter = 0

        # handle tuple conversion to discrete values
        self.p = self._param_input_conversion()

        # create list of list from the params dictionary
        self._params_temp = [list(self.p[key]) for key in self.param_keys]

        # establish max dimensions
        self.dimensions = np.prod([len(l) for l in self._params_temp])

        # apply all the set limits
        self.param_index = self._param_apply_limits()

        # create the parameter space
        self.param_space = self._param_space_creation()

        # handle the boolean limits separately
        if self.boolean_limit is not None:
            index = self._convert_lambda(self.boolean_limit)(self.param_space)
            self.param_space = self.param_space[index]

        # reset index
        self.param_index = list(range(len(self.param_index)))

    def _param_input_conversion(self):

        '''Parameters may be input as lists of single or
        multiple values (discrete values) or tuples
        (range of values). This helper checks the format of
        each input and handles it accordingly.'''

        out = {}

        # go through each parameter type
        for param in self.param_keys:

            # deal with range (tuple) values
            if isinstance(self.params[param], tuple):
                out[param] = self._param_range_expansion(self.params[param])

            # deal with range (list) values
            elif isinstance(self.params[param], list):
                out[param] = self.params[param]

        return out

    def _param_apply_limits(self):

        from talos.reducers.sample_reducer import sample_reducer

        if self.boolean_limit is not None:
            # NOTE: this is handled in __init__
            pass

        # a time limit is set
        if self.time_limit is not None:
            # NOTE: this is handled in _time_left
            pass

        # a fractional limit is set
        if self.fraction_limit is not None:
            return sample_reducer(self.fraction_limit,
                                  self.dimensions,
                                  self.random_method)

        # a round limit is set
        if self.round_limit is not None:
            return sample_reducer(self.round_limit,
                                  self.dimensions,
                                  self.random_method)

        # no limits are set
        return list(range(self.dimensions))

    def _param_range_expansion(self, param_values):

        '''Expands a range (tuple) input into discrete
        values. Helper for _param_input_conversion.
        Expects to have a input as (start, end, steps).
        '''

        start = param_values[0]
        end = param_values[1]
        steps = param_values[2]

        out = np.arange(start, end, (end - start) / steps, dtype=float)

        # inputs are all ints
        if isinstance(start, int) and isinstance(end, int):
            out = out.astype(int)
            out = np.unique(out)

        return out

    def _param_space_creation(self):

        '''Expand params dictionary to permutations

        Takes the input params dictionary and expands it to
        actual parameter permutations for the experiment.
        '''

        # handle the cases where parameter space is still large
        if len(self.param_index) > 100000:

            final_grid = list(it.product(*self._params_temp))
            out = np.array(final_grid, dtype='object')

        # handle the cases where parameter space is already smaller
        else:
            final_grid = []
            for i in self.param_index:
                p = []
                for l in reversed(self._params_temp):
                    i, s = divmod(int(i), len(l))
                    p.insert(0, l[s])
                final_grid.append(tuple(p))

            out = np.array(final_grid, dtype='object')

        return out

    def _check_time_limit(self):

        if self.time_limit is None:
            return True

        stop = datetime.strptime(self.time_limit, "%Y-%m-%d %H:%M")

        return stop > datetime.now()

    def round_parameters(self):

        # permutations remain in index
        if len(self.param_index) > 0:

            # time limit has not been met yet
            if self._check_time_limit():
                self.round_counter += 1

                # get current index
                index = self.param_index.pop(0)

                # get the values based on the index
                values = self.param_space[index]
                round_parameters = self._round_parameters_todict(values)

                # pass the parameters to Scan
                return round_parameters

        # the experiment is finished
        return False

    def _round_parameters_todict(self, values):

        round_parameters = {}

        for i, key in enumerate(self.param_keys):
            round_parameters[key] = values[i]

        return round_parameters

    def _convert_lambda(self, fn):

        '''Converts a lambda function into a format
        where parameter labels are changed to the column
        indexes in parameter space.'''

        # get the source code for the lambda function
        fn_string = inspect.getsource(fn)
        fn_string = fn_string.replace('"', '\'')

        # look for column/label names
        for i, name in enumerate(self.param_keys):
            index = ':,' + str(i)
            fn_string = fn_string.replace(name, index)

        # cleanup the string
        fn_string = fn_string.split('lambda')[1]
        fn_string = fn_string.replace('[\':', '[:')
        fn_string = fn_string.replace('\']', ']')
        fn_string = 'lambda ' + fn_string

        # pass it back as a function
        return eval(fn_string)

    def remove_is_not(self, label, value):

        '''Removes baesd on exact match but reversed'''

        col = self.param_keys.index(label)
        drop = np.where(self.param_space[:, col] != value)[0].tolist()
        self.param_index = [x for x in self.param_index if x not in drop]

    def remove_is(self, label, value):

        '''Removes based on exact match'''

        col = self.param_keys.index(label)
        drop = np.where(self.param_space[:, col] == value)[0].tolist()
        self.param_index = [x for x in self.param_index if x not in drop]

    def remove_ge(self, label, value):

        '''Removes based on greater-or-equal'''

        col = self.param_keys.index(label)
        drop = np.where(self.param_space[:, col] >= value)[0].tolist()
        self.param_index = [x for x in self.param_index if x not in drop]

    def remove_le(self, label, value):

        '''Removes based on lesser-or-equal'''

        col = self.param_keys.index(label)
        drop = np.where(self.param_space[:, col] <= value)[0].tolist()
        self.param_index = [x for x in self.param_index if x not in drop]

    def remove_lambda(self, function):

        '''Removes based on a lambda function'''

        index = self._convert_lambda(function)(self.param_space)
        self.param_space = self.param_space[index]
        self.param_index = list(range(len(self.param_space)))
