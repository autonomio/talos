from numpy import arange, unique, random, column_stack


def round_params(self):

    '''PICK PARAMETERS FOR ROUND'''

    self, _p = run_param_pick(self)
    self.params = run_param_todict(self, _p)

    return self


def run_param_pick(self):

    '''PICK SET OF PARAMETERS'''

    if self.search_method == 'random':
        _choice = random.choice(self.param_log)

    elif self.search_method == 'linear':
        _choice = min(self.param_log)

    elif self.search_method == 'reverse':
        _choice = max(self.param_log)

    self.param_log.remove(_choice)

    return self, self.param_grid[_choice]


def run_param_todict(self, params):

    _rpt_out = {}
    x = 0
    for key in self.param_dict.keys():
        _rpt_out[key] = params[x]
        x += 1

    return _rpt_out


def param_index(self):

    '''ADD PARAMETER INDEX TO PARAM GRID'''

    return column_stack([self.param_grid, self.param_log])


def param_space(self):

    '''COUNT PARAMETER COMBINATIONS'''

    _ps_out = 1

    for key in self.p.keys():

        _ps_out *= len(self.p[key])

    return _ps_out


def param_format(self):

    '''DETECT PARAM FORMAT'''

    out = {}

    for param in self.param_dict.keys():

        # for range style input
        if type(self.param_dict[param]) is type(()):
            out[param] = param_range(self.param_dict[param][0],
                                     self.param_dict[param][1],
                                     self.param_dict[param][2])
        # all other input styles
        else:
            out[param] = self.param_dict[param]

    return out


def param_range(start, end, n):

    '''PARAMETER RANGE

    param_range(0.001, 0.01, 10)

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
