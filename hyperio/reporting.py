import pandas as pd
#import astetik

from IPython.display import display, HTML
from .utils.template import df_cols

import sys
sys.path.insert(0, '/Users/mikko/Documents/GitHub/astetik')
import astetik

# these labels are used in the pretty report
report_cols = ['first_neuron',
               'batch_size',
               'round_epochs',
               'normalized_lr',
               'optimizer',
               'last_activation',
               'loss']

# text based reports are provided for these
labels = ['val_score', 'train_score']


class Reporting:

    def __init__(self, filename):

        self.filename = filename
        self.data = self._load_data()
        self.data.columns = df_cols()
        self.top_params = self._min_and_maxes(False)
        self.bottom_params = self._min_and_maxes(True)
        self.report = self._print_report()
        self.plots = astetik

    def _load_data(self):

        data = pd.read_csv(self.filename, header=None)
        data[14] = [i[1] for i in data[14].str.split()]
        data[15] = data[15].str.replace("'|>|<|\.|class|keras|optimizers|\ ", '')
        data[16] = [i[1] for i in data[16].str.split()]

        float_cols = data.select_dtypes(float).columns
        data[float_cols] = data[float_cols].round(3)

        return data

    def _min_and_maxes(self, mode):

        '''NOTE: False gives top and True bottom'''

        data = self.data.sort_values('val_score', ascending=mode)

        _mm_out = {'batch_size': (data['batch_size'].min(), data['batch_size'].max()),
                   'epochs': (data['batch_size'].min(), data['epochs'].max()),
                   'dropout': (data['dropout'].min(), data['dropout'].max()),
                   'normalized_lr': (data['normalized_lr'].min(), data['normalized_lr'].max()),
                   'embed_dims': (data['embed_dims'].min(), data['embed_dims'].max()),
                   'weight_regul': (data['weight_regul'].min(), data['weight_regul'].max()),
                   'loss': [i for i in data.loss.unique()],
                   'optimizer': [i for i in data.optimizer.unique()],
                   'last_activation': [i for i in data.last_activation.unique()]}

        return _mm_out

    def _print_report(self):

        '''PRINT PRETTY RESULT REPORT'''

        for label in labels:

            temp = self.data[self.data[label] == self.data[label].max()].head(1)

            val_score = temp[label].values * 100
            batch_size = temp['batch_size'].values
            val_peak = temp['val_peak'].values
            dropout = temp['dropout'].values
            learning_rate = temp['normalized_lr'].values
            loss = temp['loss'].values[0]
            optimizer = temp['optimizer'].values[0]
            activation = temp['last_activation'].values[0]
            first_neuron = temp['first_neuron'].values[0]

            print("\nModel with the best " + label)
            print("-" * 30)
            print("The  highest val_score is %.2f%%, \
which was achieved after %d epochs \
using %s (loss), %s (optimizer), and %s (activation). \
The batch size was %d with a dropout of %d and \
a normalized learning rate of %.2f \
and %d first layer neurons.  \n\
        " % (val_score,
             val_peak,
             loss,
             optimizer,
             activation,
             batch_size,
             dropout,
             learning_rate,
             first_neuron))

        display(HTML('<h3>highest</h3>'))
        display(self.data.sort_values('val_score', ascending=False).head(10).set_index('val_score')[report_cols])

        display(HTML('<h3>lowest</h3>'))
        display(self.data.sort_values('val_score', ascending=True).head(10).set_index('val_score')[report_cols])

        print('\n NOTE: you have more options in the Reporting object.\n')
