class AutoScan:

    def __init__(self,
                 task,
                 experiment_name,
                 max_param_values=None):

        '''Configure the `AutoScan()` experiment and then use
        the property `start` in the returned class object to start
        the actual experiment.

        `task` | str | 'binary', 'multi_class', 'multi_label', or 'continuous'
        `max_param_values` | int | Number of parameter values to be included.
                                   Note, this will only work when `params` is
                                   not passed as kwargs in `AutoScan.start`.
        '''

        self.task = task
        self.max_param_values = max_param_values
        self.experiment_name = experiment_name

    def start(self, x, y, **kwargs):

        '''Start the scan. Note that you can use `Scan()` arguments as you
        would otherwise directly interacting with `Scan()`.

        `x` | array or list of arrays | prediction features
        `y` | array or list of arrays | prediction outcome variable
        `kwargs` | arguments | any `Scan()` argument can be passed here

        '''

        import talos

        m = talos.autom8.AutoModel(self.task, self.experiment_name).model

        try:
            kwargs['params']
            scan_object = talos.Scan(x, y,
                                     model=m,
                                     experiment_name=self.experiment_name,
                                     **kwargs)
        except KeyError:
            p = talos.autom8.AutoParams(task=self.task)

            if self.max_param_values is not None:
                p.resample_params(self.max_param_values)
            params = p.params
            scan_object = talos.Scan(x=x,
                                     y=y,
                                     params=params,
                                     model=m,
                                     experiment_name=self.experiment_name,
                                     **kwargs)

        return scan_object
