class DistributeParamSpace:

    def __init__(self,
                 params,
                 param_keys,
                 random_method='uniform_mersenne',
                 fraction_limit=None,
                 round_limit=None,
                 time_limit=None,
                 boolean_limit=None,
                 machines=2):

        '''Splits ParamSpace object based on number
        of machines.

        params | object | ParamSpace class object
        machines | int | number of machines to split for

        NOTE: `Scan()` limits will not be applied if ParamSpace object
        is passed directly into `Scan()` as `params` argument so they
        should be passed directly into `DistributeParamSpace` instead.

        '''

        from talos.parameters.ParamSpace import ParamSpace

        self._params = ParamSpace(params=params,
                                  param_keys=param_keys,
                                  random_method='uniform_mersenne',
                                  fraction_limit=fraction_limit,
                                  round_limit=round_limit,
                                  time_limit=time_limit,
                                  boolean_limit=boolean_limit)

        self.machines = machines

        self.param_spaces = self._split_param_space()

    def _split_param_space(self):

        '''Takes in a ParamSpace object and splits it so that
        it can be used in DistributeScan experiments.'''

        import numpy as np
        import copy

        out = {}

        # randomly shuffle the param_space
        rand = np.random.default_rng()
        rand.shuffle(self._params.param_space, axis=0)

        # split into n arras
        param_spaces = np.array_split(self._params.param_space, self.machines)

        # remove keys to allow copy
        param_keys = self._params.param_keys
        self._params.param_keys = []

        # create the individual ParamSpace objects
        for i in range(self.machines):

            out[i] = copy.deepcopy(self._params)
            out[i].param_space = param_spaces[i]
            out[i].dimensions = len(out[i].param_space)
            out[i].param_index = list(range(out[i].dimensions))
            out[i].param_keys = param_keys

        return out
