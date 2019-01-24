from pandas import read_csv
from ..utils.connection_check import is_connected

if is_connected() is True:
    from astetik import line, hist, corr, regs, bargrid, kde, box

from ..metrics.names import metric_names


class Reporting:

    '''A suite of commands that are useful for analyzing the results
    of a completed scan, or during a scan.

    filename :: the name of the experiment log from Scan()'''

    def __init__(self, source=None):

        '''Takes as input a filename to the experiment
        log or the Scan object'''

        if isinstance(source, str):
            self.data = read_csv(source)
        else:
            self.data = source.data

    def high(self, metric='val_acc'):

        '''Returns the highest value for a given metric'''

        return max(self.data[metric])

    def rounds(self):

        '''Returns the number of rounds in the experiment'''

        return len(self.data)

    def rounds2high(self, metric='val_acc'):

        '''Returns the number of rounds it took to get to the
        highest value for a given metric.'''

        return self.data[self.data[metric] == self.data[metric].max()].index[0]

    def low(self, metric='val_acc'):

        '''Returns the minimum value for a given metric'''

        return min(self.data[metric])

    def correlate(self, metric='val_acc'):

        '''Returns a correlation table against a given metric. Drops
        all other metrics and correlates against hyperparameters only.'''

        columns = [c for c in self.data.columns if c not in metric_names()]
        out = self.data[columns]
        out.insert(0, metric, self.data[metric])
        out = out.corr()[metric]

        return out[out != 1]

    def plot_line(self, metric='val_acc'):

        '''A line plot for a given metric where rounds is on x-axis

        NOTE: remember to invoke %matplotlib inline if in notebook

        metric :: the metric to correlate against

        '''

        return line(self.data, metric)

    def plot_hist(self, metric='val_acc', bins=10):

        '''A histogram for a given metric

        NOTE: remember to invoke %matplotlib inline if in notebook

        metric :: the metric to correlate against
        bins :: number of bins to use in histogram

        '''

        return hist(self.data, metric, bins=bins)

    def plot_corr(self, metric='val_acc', color_grades=5):

        '''A heatmap with a single metric and hyperparameters.

        NOTE: remember to invoke %matplotlib inline if in notebook

        metric :: the metric to correlate against
        color_grades :: number of colors to use in heatmap'''

        cols = self._cols(metric)

        return corr(self.data[cols], color_grades=color_grades)

    def plot_regs(self, x='val_acc', y='val_loss'):

        '''A regression plot with data on two axis

        x = data for the x axis
        y = data for the y axis
        '''

        return regs(self.data, x, y)

    def plot_box(self, x, y='val_acc', hue=None):

        '''A box plot with data on two axis

        x = data for the x axis
        y = data for the y axis
        hue = data for the hue separation
        '''

        return box(self.data, x, y, hue)

    def plot_bars(self, x, y, hue, col):

        '''A comparison plot with 4 axis'''

        return bargrid(self.data,
                       x=x,
                       y=y,
                       hue=hue,
                       col=col,
                       col_wrap=4)

    def plot_kde(self, x, y=None):

        '''Kernel Destiny Estimation type histogram with
        support for 1 or 2 axis of data'''

        return kde(self.data, x, y)

    def table(self, metric='val_acc', sort_by=None, ascending=False):

        '''Shows a table with hyperparameters and a given metric

        EXAMPLE USE:

        ra1 = Reporting('diabetes_1.csv')
        ra1.table(sort_by='fmeasure_acc', ascending=False)

        PARAMS:

        metric :: accepts single column name as string or multiple in list
        sort_by :: the colunm name sorting should be based on
        ascending :: if sorting is ascending or not

        '''

        cols = self._cols(metric)

        if sort_by is None:
            sort_by = metric

        out = self.data[cols].sort_values(sort_by, ascending=ascending)

        return out

    def best_params(self, metric='val_acc', n=10, ascending=False):

        '''Get the best parameters of the experiment based on a metric.
        Returns a numpy array with the values in a format that can be used
        with the talos backend in Scan(). Adds an index as the last column.'''

        cols = self._cols(metric)
        out = self.data[cols].sort_values(metric, ascending=ascending)
        out = out.drop(metric, axis=1).head(n)
        out.insert(out.shape[1], 'index_num', range(len(out)))

        return out.values

    def _cols(self, metric):

        '''Helper to remove other than desired metric from data table'''

        cols = [col for col in self.data.columns if col not in metric_names()]

        if isinstance(metric, list) is False:
            metric = [metric]
        for i, metric in enumerate(metric):
            cols.insert(i, metric)

        # make sure only unique values in col list
        cols = list(set(cols))

        return cols
