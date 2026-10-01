from .runner import run


class UniversalExperimentLoop:
    """Limen-style Python facade over the same executor used by Scan and the CLI."""
    def __init__(self, *, sfd, data=None, search_strategy=None, pruning_strategies=None,
                 feedback_interval=100, checkpoint_interval=1, experiment_dir=None,
                 intra_callback=None, yaml_reference=None, **options):
        self._pause_requested = False
        self.sfd = sfd
        self.data = data
        self.options = dict(search_strategy=search_strategy, pruning_strategies=pruning_strategies,
                            feedback_interval=feedback_interval, checkpoint_interval=checkpoint_interval,
                            experiment_dir=experiment_dir, intra_callback=intra_callback,
                            yaml_reference=yaml_reference, **options)

    def run(self, experiment_name='experiment', n_permutations=None, **options):
        options.pop('post_processing', None)
        options.setdefault('prep_each_round', False)
        options.setdefault('pause_requested', lambda: self._pause_requested)
        self.result = run(self.sfd, self.data, experiment_name=experiment_name,
                          n_permutations=n_permutations, **{**self.options, **options})
        self.experiment_log = self.result.data
        self.models = self.result.models
        self.round_params = [record['params'] for record in self.result._records]
        return self.result

    def request_pause(self):
        self._pause_requested = True

    request_shutdown = request_pause
