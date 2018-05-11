import pandas as pd


def spear(self, metric, neg_corr=True, treshold=-.1):

    data = pd.read_csv(self.experiment_name + '.csv')

    ind = data.columns[7:]
    ind = pd.Series(list(ind), index=range(len(ind)))
    ind = ind.reset_index().set_index(0)

    data = data.copy('deep')
    if self.reduction_window == None:
        self.reduction_window = self.reduction_interval
    data = data.tail(self.reduction_window)
    metric_col = pd.DataFrame(data[metric])
    data = pd.merge(metric_col, data.iloc[:, 7:], left_index=True, right_index=True)
    correlations = data.corr('spearman')
    try:
        neg_lab = correlations[metric].dropna().sort_values(ascending=neg_corr).index[0]
    except IndexError:
        print('Reduction Failed: only NaN values')
        return "_NULL"

    dummies = pd.get_dummies(data[neg_lab])
    merged = pd.merge(metric_col, dummies, left_index=True, right_index=True)
    corr = merged.corr()[metric].sort_values(ascending=neg_corr)

    if corr[0] < treshold:
        return (corr.index[0], int(ind.loc[neg_lab].values))
    else:
        return "_NULL"


def spear_reducer(self):

    to_drop = spear(self, self.reduction_metric)

    # if a value have been returned, proceed with dropping
    if to_drop != "_NULL":
        index_of_drops = self.param_grid[self.param_grid[:, to_drop[1]] == to_drop[0]][:,-1]
        self.param_log = list(set(self.param_log).difference(set(index_of_drops)))

    return self
