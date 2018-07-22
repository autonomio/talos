import pandas as pd

from IPython.display import display, HTML

import astetik


class Reporting:
    """Output table of the Scan execution. Takes as an argument a string
    of the file name of the execution set during the call to Scan()."""

    def __init__(self, filename):

        self.filename = filename
        self.data = self._load_data()

        self.top_params = self._min_and_maxes(False)
        self.report = self._print_report()
        self.plots = astetik

    def _load_data(self):
        """Loads the saved csv data file from the execution."""

        data = pd.read_csv(self.filename)
        # cleanes up the function/class name artifacts
        for col in data.columns:
            try:
                if data[col][0].startswith('<'):
                    data[col] = data[col].str.replace('keras.optimizers.', '')\
                        .str.replace("'|\.", ' ')
                    data[col] = [i[1] for i in data[col].str.split()]
            except AttributeError:
                pass

        float_cols = data.select_dtypes(float).columns
        data[float_cols] = data[float_cols].round(3)

        return data

    def _min_and_maxes(self, mode):
        """Get the best and worst parameter data points, sorted by validation
        accuracy."""

        # TODO: validation accuracy may not be the best metric to use
        # add option to implement other metrics

        mins = pd.DataFrame(self.data.sort_values('val_acc').tail(10).min())
        maxs = pd.DataFrame(self.data.sort_values('val_acc').tail(10).max())
        min_max = pd.merge(mins, maxs, left_index=True,
                           right_index=True).tail(-9)
        min_max.columns = ['min', 'max']

        return min_max

    def _print_report(self):
        """Print the report. Depending on the notebook being used, the format
        may be distorted, in which case pandas can be used directly."""

        # TODO: implement the alternative printing method

        display(HTML('<h3>highest</h3>'))
        display(self.data.sort_values('val_acc', ascending=False)
                .head(10).set_index('val_acc').iloc[:, 6:])

        display(HTML('<h3>lowest</h3>'))
        display(self.data.sort_values('val_acc', ascending=True)
                .head(10).set_index('val_acc').iloc[:, 6:])

        print('\n NOTE: you have more options in the Reporting object.\n')
