from .experiment_log import _callback_base


class TrainingPlot:
    def __new__(cls, backend='keras', **kwargs):
        base = _callback_base(backend)
        class PlotCallback(base):
            def __init__(self):
                super().__init__()
                self.history = {}

            def on_train_begin(self, logs=None):
                from matplotlib import pyplot as plt
                self.figure, self.axes = plt.subplots()
                self.history = {}

            def on_epoch_end(self, epoch, logs=None):
                for key, value in (logs or {}).items():
                    self.history.setdefault(key, []).append(value)
                self.axes.clear()
                for key, values in self.history.items():
                    self.axes.plot(range(1, len(values) + 1), values, label=key)
                self.axes.set_xlabel('Epoch')
                self.axes.legend()
                self.figure.canvas.draw_idle()
        return PlotCallback()
