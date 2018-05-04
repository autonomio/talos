import numpy as np
import datetime as dt
import pandas as pd

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dropout, Dense
from keras.callbacks import EarlyStopping

from keras.utils import to_categorical
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

# Activations
from keras.activations import elu, selu, relu, hard_sigmoid, linear, sigmoid, softmax, softplus, softsign, tanh

# Optimizers
from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Adamax, RMSprop, Nadam

# Losses
from keras.losses import binary_crossentropy, kullback_leibler_divergence
from keras.losses import categorical_hinge, categorical_crossentropy, sparse_categorical_crossentropy
from keras.losses import cosine, cosine_proximity, hinge, logcosh, mae, mape, mse, msle, poisson, squared_hinge


class Hyperio:
            
    global self
    
    def __init__(self, x, y, params, experiment_name, model,
                 val_split=.3, search_method='random', reduction_method=None,
                 grid_downsample=None, early_stopping=None):
        
        self.model = model
        self.experiment_name = experiment_name
        self.param_dict = params
        self.search_method = search_method
        self.reduction_method = None
        self.grid_downsample = grid_downsample
        self.early_stopping = early_stopping
        self.val_split = val_split
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
        self.x_train, self.y_train, self.x_val, self.y_val = self._val_split()
        
        # run the scan
        self.result = []
        self._null = self._hyper_run()
        
        # get the results ready
        self._null = self._result_todf()
        
    
    def _save_result(self):
        
        np.savetxt(self.experiment_name + '.csv', self.result, fmt='%s', delimiter=',')
        
        
    def _result_todf(self):
        
        self.result = pd.DataFrame(self.result)
        self.result.columns = ['train_peak', 'test_peak', 'train_acc', 'val_acc',
                               'train_loss', 'val_loss', 'train_score', 'val_score',
                               'batch_size', 'epochs','dropout','lr','loss',
                               'optimizer','activation', 'weight_regulurizer','emb_output_dims']
        
    def _run_round_params(self):
        
        _p = self._run_param_pick()
        _p = self._run_param_todict(_p)
        
        self.params = _p
        
        
    def _model(self):
        
        return self.model(self.x_train, 
                          self.y_train,
                          self.x_val,
                          self.y_val,
                          self.params)
        
    def _hyper_run(self):
        
        '''RUNTIME'''
        
        for i in range(len(self.param_grid)):
            
            self._run_round_params()
            _hr_out = self._model()
            _hr_out = self._run_round_results(_hr_out)
            self.result.append(_hr_out)
            self._save_result()
            self._estimator()
            
            self.round_counter +=1
        
    def _val_split(self):

        len_x = len(self.x)
        limit = int(len_x * (1 - self.val_split))

        x_train = self.x[:limit] 
        y_train = self.y[:limit]

        x_test = self.x[limit:]
        y_test = self.y[limit:]

        if len(x_train) + len(x_test) == len(x):
            return x_train, y_train, x_test, y_test
        else:
            return print("Something went wrong")
        
    
    def _early_stopper(self, epochs):

        '''EARLY STOP CALLBACK

        Helps prevent wasting time when loss is not becoming
        better. Offers two pre-determined settings 'moderate'
        and 'strict' and allows input of list with two values:

        min_delta = the limit for change at which point flag is raised

        patience = the number of epochs before termination from flag

        '''



        if self.early_stopping == None:
            _es_out = None

        elif self.early_stopping == 'moderate':
            _es_out = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    min_delta=0,
                                                    patience=epochs / 10,
                                                    verbose=0, mode='auto')
        elif self.early_stopping == 'strict':
            _es_out = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    min_delta=0,
                                                    patience=2,
                                                    verbose=0, mode='auto')
        elif type(self.early_stopping) == type([]):
            _es_out = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                    min_delta=self.early_stopping[0],
                                                    patience=self.early_stopping[1],
                                                    verbose=0, mode='auto')
        return _es_out    


        
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
        
        Note that you have to change dimensions 
        for the reshape after adding new params.
        
        '''
        
        _pg_out = np.meshgrid(self.p['lr'],
                              self.p['batch_size'],
                              self.p['epochs'],
                              self.p['dropout'],
                              self.p['optimizer'],
                              self.p['loss'],
                              self.p['activation'],
                              self.p['weight_regulizer'],
                              self.p['emb_output_dims'])

        return np.array(_pg_out).T.reshape(-1, 9)

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

            # for range
            if type(self.param_dict[param]) is type(()):
                out[param] = self._param_range(p[param][0], p[param][1], p[param][2])

            # all other cases
            else:
                out[param] = self.param_dict[param]
            
        return out
        
    def _param_range(self, start, end, n):

        '''PARAMETER RANGE 

        param_range(0.001, 0.01, 10)

        '''

        out =  np.arange(start, end, (end - start) / n, dtype=float)

        if type(start) == int and type(end) == int:
            out = out.astype(int)
            out = np.unique(out)

        return out


    def _run_round_results(self, out):

        train_peak = np.argpartition(np.array(out.history['acc']) - np.array(out.history['loss']), self.params['epochs']-1)[-1]
        test_peak = np.argpartition(np.array(out.history['val_acc']) - np.array(out.history['val_loss']), self.params['epochs']-1)[-1]

        train_acc = np.array(out.history['acc'])[train_peak]
        train_loss = np.array(out.history['loss'])[train_peak]
        train_score = train_acc - train_loss

        val_acc = np.array(out.history['val_acc'])[train_peak]
        val_loss = np.array(out.history['val_loss'])[train_peak]
        val_score = val_acc - val_loss

        rr_out = [train_peak, test_peak,
                  train_acc, val_acc,
                  train_loss, val_loss,
                  train_score, val_score,
                  self.params['batch_size'], self.params['epochs'],
                  self.params['dropout'], self.params['lr'],
                  self.params['loss'], self.params['optimizer'],
                  self.params['activation'], self.params['weight_regulizer'],
                  self.params['emb_output_dims']]

        return rr_out
