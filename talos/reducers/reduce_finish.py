def reduce_finish(self):

    '''Takes input from a Reducer in form of a tuple
    where the values the hyperparamater name and the
    value to drop. Returns self with a modified param_log.'''

    # get the column index

    to_remove_col = self.param_reference[self._reduce_keys[1]]

    value_to_remove = self._reduce_keys[0]

    # pick the index numbers for dropping available permutations
    indexs_to_drop = self.param_grid[self.param_grid[:, to_remove_col] == value_to_remove][:,-1]

    # drop the index numbers
    self.param_log = list(set(self.param_log).difference(set(indexs_to_drop)))

    return self
