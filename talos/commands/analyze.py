"""Legacy experiment analytics using current pandas and optional matplotlib."""
import json
from pathlib import Path
import pandas as pd


class Analyze:
    def __init__(self, source=None):
        self._parameter_columns = getattr(source, 'parameter_columns', None)
        params = getattr(source, 'params', None)
        if isinstance(source, (str, bytes, Path)):
            self.data = pd.read_csv(source)
            metadata_path = Path(source).resolve().parent / 'metadata.json'
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text())
                self._parameter_columns = metadata.get('parameter_columns')
                params = metadata.get('params')
        else:
            self.data = source if isinstance(source, pd.DataFrame) else source.data
        if not isinstance(params, dict):
            params = getattr(params, 'params', None)
        self._parameter_keys = list(params) if isinstance(params, dict) else None
        if self._parameter_keys is None and self._parameter_columns is not None:
            self._parameter_keys = list(self._parameter_columns)

    def high(self, metric):
        return self.data[metric].max()

    def low(self, metric):
        return self.data[metric].min()

    def rounds(self):
        return len(self.data)

    def rounds2high(self, metric):
        return self.data[metric].idxmax()

    def correlate(self, metric, exclude, method='pearson'):
        columns = [name for name in self.data if name not in exclude]
        return self.data[columns].corr(method=method, numeric_only=True)[metric].drop(metric)

    def _cols(self, metric, exclude):
        metrics = metric if isinstance(metric, list) else [metric]
        return list(dict.fromkeys(metrics + [name for name in self.data if name not in exclude]))

    def table(self, metric, exclude=None, sort_by=None, ascending=False):
        return self.data[self._cols(metric, exclude or [])].sort_values(sort_by or metric, ascending=ascending)

    def best_params(self, metric, exclude, n=10, ascending=False):
        if self._parameter_columns is not None:
            columns = [self._parameter_columns.get(key, key) for key in self._parameter_keys]
            columns = [column for column in columns if column not in (exclude or [])]
            out = self.data.sort_values(metric, ascending=ascending, kind='stable')[columns].head(n).copy()
        else:
            out = self.table(metric, exclude, ascending=ascending).drop(columns=metric).head(n).copy()
        out['index_num'] = range(len(out))
        return out.to_numpy()

    @staticmethod
    def _axes():
        from matplotlib import pyplot as plt
        return plt.subplots()[1]

    def plot_line(self, metric):
        ax = self._axes()
        self.data[metric].plot(ax=ax)
        ax.set_xlabel('Round')
        ax.set_ylabel(metric)
        return ax

    def plot_hist(self, metric, bins=10):
        ax = self._axes()
        self.data[metric].plot.hist(bins=bins, ax=ax)
        ax.set_xlabel(metric)
        return ax

    def plot_corr(self, metric, exclude, color_grades=5):
        from matplotlib import pyplot as plt
        columns = self._cols(metric, exclude)
        corr = self.data[columns].corr(numeric_only=True)
        ax = self._axes()
        picture = ax.imshow(corr, vmin=-1, vmax=1, cmap=plt.get_cmap('coolwarm', color_grades))
        ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
        ax.set_yticks(range(len(corr)), corr.columns)
        ax.figure.colorbar(picture, ax=ax)
        return ax

    def plot_regs(self, x, y):
        import numpy as np
        ax = self._axes()
        pairs = self.data[[x, y]].dropna()
        ax.scatter(pairs[x], pairs[y])
        if len(pairs) > 1 and pairs[x].nunique() > 1:
            fit = np.polyfit(pairs[x], pairs[y], 1)
            xs = np.sort(pairs[x].to_numpy())
            ax.plot(xs, np.polyval(fit, xs))
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        return ax

    def plot_box(self, x, y, hue=None):
        ax = self._axes()
        by = [x, hue] if hue else x
        self.data.boxplot(column=y, by=by, ax=ax)
        return ax

    def plot_bars(self, x, y, hue, col):
        from matplotlib import pyplot as plt
        groups = list(self.data.groupby(col, dropna=False))
        figure, axes = plt.subplots(1, len(groups), squeeze=False, figsize=(5 * len(groups), 4))
        for ax, (name, group) in zip(axes[0], groups):
            group.pivot_table(index=x, columns=hue, values=y).plot.bar(ax=ax)
            ax.set_title(str(name))
        return figure

    def plot_kde(self, x, y=None):
        ax = self._axes()
        if y is None:
            self.data[x].plot.kde(ax=ax)
        else:
            import numpy as np
            from scipy.stats import gaussian_kde
            pairs = self.data[[x, y]].apply(pd.to_numeric, errors='coerce').dropna().to_numpy(dtype=float)
            pairs = pairs[np.isfinite(pairs).all(axis=1)]
            if len(pairs) < 3 or np.any(np.ptp(pairs, axis=0) == 0):
                raise ValueError('Bivariate KDE requires at least three finite pairs with variation on both axes.')
            density = gaussian_kde(pairs.T)
            low, high = pairs.min(axis=0), pairs.max(axis=0)
            padding = (high - low) * .1
            grid_x, grid_y = np.mgrid[low[0] - padding[0]:high[0] + padding[0]:100j,
                                     low[1] - padding[1]:high[1] + padding[1]:100j]
            estimated = density(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
            ax.contourf(grid_x, grid_y, estimated, levels=12)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        return ax
