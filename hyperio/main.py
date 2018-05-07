import numpy as np
import datetime as dt
import pandas as pd
import itertools

import re
from keras import backend as K

from .utils.template import df_cols, value_cols


class Hyperio:

    global self

    def __init__(self, x, y, params, dataset_name, experiment_no, model,
                 val_split=.3, shuffle=True, search_method='random',
                 reduction_method=None, reduction_interval=100,
                 reduction_window=None,
                 grid_downsample=None, hyperio_log_name='hyperio.log',
                 debug=False):

        # experiment name
        self.dataset_name = dataset_name
        self.experiment_no = experiment_no
        self.experiment_name = dataset_name + '_' + experiment_no

        # logfile initialization
        if debug == True:
            self.logfile = open('hyperio.debug.log', 'a')
        else:
            self.logfile_name = hyperio_log_name
            self.logfile = open(self.logfile_name, 'a')

        # load params dictionary and model
        self.model = model
        self.param_dict = params

        # load input parameters
        self.search_method = search_method
        self.reduction_method = reduction_method
        self.reduction_interval = reduction_interval
        self.reduction_window = reduction_window
        self.grid_downsample = grid_downsample
        self.val_split = val_split
        self.shuffle = shuffle
        self.df_col_list = df_cols()

        # prepare the parameter search boundary
        self.p = self._param_format()
        self.combinations = self._param_space()
        self.param_grid = self._param_grid()
        self.param_grid = self._grid_downsampling()
        self.param_log = list(range(len(self.param_grid)))
        self.param_grid = self._param_index()
        self.round_counter = 0

        # prepare data
        self.x = x
        self.y = y
        self._null = self._val_split()

        # create data related log data
        self._data_len = len(self.x)
        self._null = self._prediction_type()

        # run the scan
        self.result = []

        while len(self.param_log) != 0:
            self._null = self._hyper_run()

        # get the results ready
        self._null = self._result_todf()

        # close the log file
        self._null = self.logfile.close()
        print('Scan Finished!')

    # THE MAIN RUNTIME FUNCTION STARTS
    # --------------------------------
    def _hyper_run(self):

        '''RUNTIME'''

        self._run_round_params()    # this creates the params round
        _hr_out = self._model()
        _hr_out = self._run_round_results(_hr_out)
        self._run_write_log()
        self.result.append(_hr_out)
        self._save_result()

        # this is for the first round only
        self._estimator()

        # reducing algorithsm
        if (self.round_counter + 1) % self.reduction_interval == 0:

            if self.reduction_method == 'spear':
                self.score_based_reducer()

        # prevent Tensorflow memory leakage
        K.clear_session()

        # add one to the total round counter
        self.round_counter += 1

    # ------------------------------
    # THE MAIN RUNTIME FUNCTION ENDS

    def corr_reducer(self, metric, neg_corr=True, treshold=-.1):

        data = pd.read_csv(self.experiment_name + '.csv')

        ind = data.columns[9:]
        ind = pd.Series(list(ind), index=range(len(ind)))
        ind = ind.reset_index().set_index(0)

        data = data.copy('deep')
        if self.reduction_window == None:
            self.reduction_window = self.reduction_interval
        data = data.tail(self.reduction_window)
        metric_col =  pd.DataFrame(data[metric])
        data = pd.merge(metric_col, data.iloc[:, 9:], left_index=True, right_index=True)
        correlations = data.corr('spearman')

        neg_lab = correlations[metric].dropna().sort_values(ascending=neg_corr).index[0]

        dummies = pd.get_dummies(data[neg_lab])
        merged = pd.merge(metric_col, dummies, left_index=True, right_index=True)
        corr = merged.corr()[metric].sort_values(ascending=neg_corr)

        if corr[0] < treshold:
            return (corr.index[0], int(ind.loc['lr'].values))
        else:
            return "_NULL"

    def score_based_reducer(self):

        to_drop = self.corr_reducer('val_score')

        # if a value have been returned, proceed with dropping
        if to_drop != "_NULL":
            index_of_drops = self.param_grid[self.param_grid[:, to_drop[1]] == to_drop[0]][:,-1]
            self.param_log = list(set(self.param_log).difference(set(index_of_drops)))



        # otherwise do nothing

    def _run_round_params(self):

        '''PICK PARAMETERS FOR ROUND'''

        _p = self._run_param_pick()
        _p = self._run_param_todict(_p)

        self.params = _p

    def _model(self):

        '''RUNDS THE USERS MODEL

        This is loaded from the actual user Hyperio call from
        model= parameter.

        '''

        return self.model(self.x_train,
                          self.y_train,
                          self.x_val,
                          self.y_val,
                          self.params)

    def _val_split(self):

        '''VALIDATION SPLIT OF X AND Y
        Based on the Hyperio() parameter val_split
        both 'x' and 'y' are split.

        '''

        if self.shuffle == True:
            self._random_shuffle()

        len_x = len(self.x)
        limit = int(len_x * (1 - self.val_split))

        self.x_train = self.x[:limit]
        self.y_train = self.y[:limit]

        self.x_val = self.x[limit:]
        self.y_val = self.y[limit:]

    def _random_shuffle(self):

        random_index = np.arange(len(self.x))
        np.random.shuffle(random_index)

        self.x = self.x[random_index]
        self.y = self.y[random_index]

    def _grid_downsampling(self):

        if self.grid_downsample is None:

            return self.param_grid

        else:
            np.random.shuffle(self.param_grid)
            _d_out = int(len(self.param_grid) * self.grid_downsample)

            return self.param_grid[:_d_out]

    def _run_param_todict(self, params):

        _rpt_out = {}
        x = 0
        for key in self.param_dict.keys():
            _rpt_out[key] = params[x]
            x += 1

        return _rpt_out

    def _run_param_pick(self):

        '''PICK SET OF PARAMETERS'''

        if self.search_method == 'random':
            _choice = np.random.choice(self.param_log)

        elif self.search_method == 'linear':
            _choice = self.param_log.min()

        elif self.search_method == 'reverse':
            _choice = self.param_log.max()

        self.param_log.remove(_choice)

        return self.param_grid[_choice]

    def _param_index(self):

        '''ADD PARAMETER INDEX TO PARAM GRID'''

        return np.column_stack([self.param_grid, self.param_log])

    def _estimator(self):

        '''ESTIMATE DURATION'''

        if self.round_counter == 0:

            start = dt.datetime.now()
            self._model()
            end = dt.datetime.now()

            total = (end - start).seconds
            total = total * len(self.param_grid)

            print("%d scans will take roughly %d seconds" % (len(self.param_grid), total))

    def _param_grid(self):

        '''CREATE THE PARAMETER PERMUTATIONS
        '''

        ls = [list(self.p[key]) for key in self.p.keys()]
        _pg_out = np.array(list(itertools.product(*ls)))

        return _pg_out


    def _param_space(self):

        '''COUNT PARAMETER COMBINATIONS'''

        _ps_out = 1

        for key in self.p.keys():

            _ps_out *= len(self.p[key])

        return _ps_out

    def _param_format(self):

        '''DETECT PARAM FORMAT'''

        out = {}

        for param in self.param_dict.keys():

            # for range style input
            if type(self.param_dict[param]) is type(()):
                out[param] = self._param_range(self.param_dict[param][0],
                                               self.param_dict[param][1],
                                               self.param_dict[param][2])
            # all other input styles
            else:
                out[param] = self.param_dict[param]

        return out

    def _param_range(self, start, end, n):

        '''PARAMETER RANGE

        param_range(0.001, 0.01, 10)

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

    def _prediction_type(self):

        try:
            y_cols = self.y.shape[1]
        except IndexError:
            y_cols = 1
        y_max = self.y.max()
        y_uniques = len(np.unique(self.y))

        if y_cols > 1:
            self._y_type = 'category'
            self._y_range = y_cols
            self._y_format = 'onehot'
        else:
            if y_max == 1:
                self._y_type = 'binary'
                self._y_range = y_cols
                self._y_format = 'single'
            elif np.mean(self.y) == np.median(self.y):
                self._y_type = 'category'
                self._y_range = y_uniques
                self._y_format = 'single'
            else:
                self._y_type = 'continuous'
                self._y_num = self.y.max() - self.y.min()
                self._y_format = 'single'

    # hyperio log writing functions start
    # -----------------------------------------------
    def _clean_dict(self):

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
                elif type(self.params[key]) == type(1):
                    _cd_out[key] = int(s)
                else:
                    _cd_out[key] = s

        return _cd_out

    def _dict_tostr(self, d):

        '''PARSES THE LOG ENTRY'''

        s = "','".join(str(d[key]) for key in d.keys())
        s = re.sub("'", "", s, count=9)
        s = re.sub("$", "'", s)

        # add some values
        s += ',' + str(round(self._val_score, 3))
        s += ',' + str(self._round_epochs)
        s += ',' + self._y_type
        s += ',' + str(self._y_range)
        s += ',' + self._y_format
        s += ',' + str(self.val_split)
        s += ',' + self.dataset_name

        return s

    def _run_write_log(self):

        '''this is operated from _hyper_run'''

        _wt_out = self._clean_dict()
        _wt_out = self._dict_tostr(_wt_out)
        self.logfile.write(_wt_out + '\n')

    # -------------------------------------------
    # log writing functions end

    # results processing and saving starts
    # ------------------------------------------

    def _run_round_results(self, out):

        '''THE MAIN FUNCTION FOR CREATING RESULTS FOR EACH ROUND

        NOTE: The epoch level data will be dropped here each round.

        '''
        round_epochs = len(out.history['acc'])

        t_t = np.array(out.history['acc']) - np.array(out.history['loss'])
        v_t = np.array(out.history['val_acc']) - np.array(out.history['val_loss'])

        train_peak = np.argpartition(t_t, round_epochs-1)[-1]
        val_peak = np.argpartition(v_t, round_epochs-1)[-1]

        train_acc = np.array(out.history['acc'])[train_peak]
        train_loss = np.array(out.history['loss'])[train_peak]
        train_score = train_acc - train_loss

        val_acc = np.array(out.history['val_acc'])[val_peak]
        val_loss = np.array(out.history['val_loss'])[val_peak]
        val_score = val_acc - val_loss

        # this is for the log
        self._val_score = val_score
        self._round_epochs = round_epochs

        # if the round counter is zero, just output header
        if self.round_counter == 0:
            _rr_out = value_cols()
            [_rr_out.append(key) for key in self.params.keys()]
            return _rr_out

        # otherwise proceed to create the value row
        _rr_out = []
        for val in value_cols():
            _rr_out.append(locals()[val])

        # this is based on columns defined in template.py
        for key in self.params.keys():
            _rr_out.append(self.params[key])

        return _rr_out

    def _save_result(self):
        '''SAVES THE RESULTS/PARAMETERS TO A CSV SPECIFIC TO THE EXPERIMENT'''

        np.savetxt(self.experiment_name + '.csv',
                   self.result,
                   fmt='%s',
                   delimiter=',')

    def _result_todf(self):
        '''ADDS A DATAFRAME VERSION OF THE RESULTS TO THE CLASS OBJECT'''

        self.result = pd.DataFrame(self.result)
        self.result.columns = self.result.iloc[0]
        self.result = self.result.drop(0)

    # -------------------------------------------
    # results processing and saving ends
