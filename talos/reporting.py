import pandas as pd

from IPython.display import display, HTML

import astetik


class Reporting:

    def __init__(self, filename):

        self.filename = filename
        self.data = self._load_data()

        self.top_params = self._min_and_maxes(False)
        self.report = self._print_report()
        self.plots = astetik

    def _load_data(self):

        data = pd.read_csv(self.filename)
        # cleanes up the function/class name artifacts
        for col in data.columns:
            try:
                if data[col][0].startswith('<'):
                    data[col] = data[col].str.replace("'|\.",' ')
                    data[col] = [i[1] for i in data[col].str.split()]
            except AttributeError:
                pass

        float_cols = data.select_dtypes(float).columns
        data[float_cols] = data[float_cols].round(3)

        return data

    def _min_and_maxes(self, mode):

        mins = pd.DataFrame(self.data.sort_values('val_acc').tail(10).min())
        maxs = pd.DataFrame(self.data.sort_values('val_acc').tail(10).max())
        min_max = pd.merge(mins, maxs, left_index=True, right_index=True).tail(-9)
        min_max.columns = ['min', 'max']

        return min_max

    def _print_report(self):

        '''PRINT PRETTY RESULT REPORT'''

        display(HTML('<h3>highest</h3>'))
        display(self.data.sort_values('val_acc', ascending=False).head(10).set_index('val_acc').iloc[:,7:])

        display(HTML('<h3>lowest</h3>'))
        display(self.data.sort_values('val_acc', ascending=True).head(10).set_index('val_acc').iloc[:,7:])

        print('\n NOTE: you have more options in the Reporting object.\n')
