"""Legacy training callbacks use the shared executor and preserve checkpointed row selection."""
from types import SimpleNamespace
from talos.experiment.runner import RunResult


class Scan(RunResult):
    def __init__(self, x, y, params, model, experiment_name, x_val=None, y_val=None,
                 val_split=.3, multi_input=False, random_method='uniform_mersenne', seed=None,
                 performance_target=None, fraction_limit=None, round_limit=None, time_limit=None,
                 boolean_limit=None, reduction_method=None, reduction_interval=50,
                 reduction_window=20, reduction_threshold=.2, reduction_metric='val_acc',
                 minimize_loss=False, disable_progress_bar=False, print_params=False,
                 clear_session=True, save_weights=True, save_models=False, **options):
        from talos.parameters.ParamSpace import ParamSpace
        from talos.parameters._resume import _resume_initial_state
        from talos.utils.validation_split import validation_split
        from talos.experiment.runner import run

        if not callable(model):
            raise TypeError('model must be a callable training function')
        values = {key: value for key, value in locals().items() if key not in ('self', 'options')}
        self.__dict__.update(values)
        self.custom_val_split = x_val is not None or y_val is not None
        if (x_val is None) != (y_val is None):
            raise ValueError('x_val and y_val must be supplied together')
        validation_split(self)
        if isinstance(params, dict):
            self.param_object = ParamSpace(params, list(params), random_method=random_method,
                                          fraction_limit=fraction_limit, round_limit=round_limit,
                                          time_limit=time_limit, boolean_limit=boolean_limit, seed=seed,
                                          _initial_state=_resume_initial_state(options, list(params)))
        elif isinstance(params, ParamSpace):
            self.param_object = params
        else:
            raise TypeError('params must be a dictionary or ParamSpace')
        self._param_dict_keys = list(self.param_object.param_keys)
        self.round_history = []
        self.result = []
        self.first_round = True
        self.saved_models = []
        self.saved_weights = []

        def prep(data, round_params):
            return data

        def legacy_model(data, round_params):
            return self.model(self.x_train, self.y_train, self.x_val, self.y_val, round_params)

        sfd = SimpleNamespace(params=lambda: self.param_object.params, prep=prep, model=legacy_model)
        data = {'x_train': self.x_train, 'y_train': self.y_train, 'x_val': self.x_val, 'y_val': self.y_val}
        objective = options.pop('objective', {'metric': reduction_metric, 'direction': 'min' if minimize_loss else 'max'})
        outcome = run(sfd, data, params=params, experiment_name=experiment_name,
                      seed=seed, progress_bar=not disable_progress_bar,
                      save_models=save_models, save_weights=save_weights, clear_session=clear_session,
                      print_params=print_params, time_limit=time_limit, objective=objective,
                      legacy_context=self, source_model=model, **options)
        original_x, original_y = self.x, self.y
        self.__dict__.update(outcome.__dict__)
        self.x, self.y = original_x, original_y
        self.params = params
        self.details['random_method'] = self.random_method
        self.details['reduction_method'] = self.reduction_method
        self.details['reduction_metric'] = self.reduction_metric
        self.details['reduction_interval'] = self.reduction_interval
        self.details['reduction_window'] = self.reduction_window
        self.details['reduction_threshold'] = self.reduction_threshold
        self.details['minimize_loss'] = self.minimize_loss
        self.details['x_shape'] = getattr(x, 'shape', 'multi-input')
        self.details['y_shape'] = getattr(y, 'shape', 'multi-output')

    def best_model(self, metric='val_acc', asc=False, saved=False, custom_objects=None, **kwargs):
        from talos.utils.best_model import activate_model, best_model
        if metric is None:
            metric, asc = self._objective(metric, asc)
        return activate_model(self, best_model(self, metric, asc), saved, custom_objects, **kwargs)

    def evaluate_models(self, *args, **kwargs):
        from talos.commands.evaluate import evaluate_models
        return evaluate_models(self, *args, **kwargs)
