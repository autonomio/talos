import re


def clean_dict(self):

    '''this is operated from _write_tolog'''

    _cd_out = {}

    for key in self.params.keys():

            s = str(self.params[key])
            if s.startswith('<class'):
                _cd_out[key] = s.strip("<|>|\'").split('.')[2]
            elif s.startswith('<function'):
                _cd_out[key] = s.split()[1]
            elif s.startswith('None'):
                _cd_out[key] = None
            elif len(s.split('.')) > 1:
                _cd_out[key] = self.params[key]
            elif isinstance(self.params[key], int):
                _cd_out[key] = int(s)
            else:
                _cd_out[key] = s

    return _cd_out


def dict_tostr(self, d):

    '''PARSES THE LOG ENTRY'''

    s = "','".join(str(d[key]) for key in d.keys())
    s = re.sub("'", "", s, count=9)
    s = re.sub("$", "'", s)

    # add some values
    try:
        s += ',' + str(round(self._val_score, 3))
    except TypeError:
        s += ',' + self._val_score
        s += ',' + str(self._round_epochs)
        s += ',' + self.shape
        s += ',' + str(self._y_range)
        s += ',' + self._y_format
        s += ',' + str(self.val_split)
        s += ',' + self.dataset_name

    return s


def write_log(self):

    '''this is operated from _hyper_run'''

    _wt_out = clean_dict(self)
    _wt_out = dict_tostr(self, _wt_out)
    self.logfile.write(_wt_out + '\n')


def debug_logging(self):

    if self.debug:
        self.logfile = open('talos.debug.log', 'a')
    else:
        self.logfile_name = self.talos_log_name
        self.logfile = open(self.logfile_name, 'a')

    return self
