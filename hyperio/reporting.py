import pandas as pd

from IPython.display import display, HTML
from .utils.template import rep_cols

import astetik

# these labels are used in the pretty report
report_cols = rep_cols()

# text based reports are provided for these
labels = ['val_score', 'train_score']


class Reporting:

    def __init__(self, filename):

        self.filename = filename
        self.data = self._load_data()

        self.top_params = self._min_and_maxes(False)
        self.report = self._print_report()
        self.plots = astetik

    def _load_data(self):

        data = pd.read_csv(self.filename)
        data['optimizer'] = data['optimizer'].str.replace("'|>|<|\.|class|keras|optimizers|\ ", '')

        for col in ['loss', 'activation', 'last_activation']:
            try:
                data[col] = [i[1] for i in data[col].str.split()]
            except (KeyError, IndexError):
                pass

        float_cols = data.select_dtypes(float).columns
        data[float_cols] = data[float_cols].round(3)

        return data

    def _min_and_maxes(self, mode):

        mins = pd.DataFrame(self.data.min())
        maxs = pd.DataFrame(self.data.max())
        min_max = pd.merge(mins, maxs, left_index=True, right_index=True).tail(-9)
        min_max.columns = ['min', 'max']

        return min_max

    def _print_report(self):

        '''PRINT PRETTY RESULT REPORT'''

        display(HTML('<h3>highest</h3>'))
        display(self.data.sort_values('val_score', ascending=False).head(10).set_index('val_score')[report_cols])

        display(HTML('<h3>lowest</h3>'))
        display(self.data.sort_values('val_score', ascending=True).head(10).set_index('val_score')[report_cols])

        print('\n NOTE: you have more options in the Reporting object.\n')
