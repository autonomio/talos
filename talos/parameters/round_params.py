from numpy import random


def round_params(self):

    '''Picks the paramaters for a round based on the available
    paramater permutations using the param_log index'''

    # pick the permutation for the round
    if self.search_method == 'random':
        _choice = random.choice(self.param_log)

    elif self.search_method == 'linear':
        _choice = min(self.param_log)

    elif self.search_method == 'reverse':
        _choice = max(self.param_log)

    # remove the current choice from permutations
    self.param_log.remove(_choice)

    # create a dictionary for the current round
    _round_params_dict = {}
    x = 0
    for key in self.param_reference.keys():
        _round_params_dict[key] = self.param_grid[_choice][x]
        x += 1

    return _round_params_dict
