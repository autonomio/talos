"""Adapt legacy parameter selection to the checkpointed search queue."""

import itertools

from talos.experiment.serialization import content_hash

from .search_strategy import SearchStrategy


class LegacyStrategy(SearchStrategy):
    """Talos' pending row facade, iterated by the same Mutable Search Queue."""

    def __init__(self, facade, domain, *, seed=None):
        self.facade = facade
        super().__init__(domain, seed=seed)
        self._known_values = {key: {content_hash(value) for value in values} for key, values in domain.params.items()}

    @property
    def is_finite(self):
        return True

    @property
    def trial_namespace(self):
        return self.facade.shard_namespace

    def __next__(self):
        if not self.facade._check_time_limit():
            raise StopIteration
        while self.facade.param_index:
            index = self.facade.param_index.pop(0)
            values = self.facade.param_space[index]
            combo = self.facade._round_parameters_todict(values)
            if self.domain.is_valid_combination(combo):
                self.facade.round_counter += 1
                self._generated_count += 1
                return combo
        raise StopIteration

    def remaining_count(self):
        return sum(self.domain.is_valid_combination(self.facade._round_parameters_todict(self.facade.param_space[index])) for index in self.facade.param_index)

    def get_state(self):
        return {'pending': list(self.facade.param_index),
                'shard_namespace': self.facade.shard_namespace,
                'shard_id': self.facade.shard_id,
                'param_space': self.facade.param_space,
                'round_counter': self.facade.round_counter,
                'generated_count': self._generated_count,
                'known_values': {key: sorted(values) for key, values in self._known_values.items()}}

    def set_state(self, state):
        import numpy as np
        self.facade.param_space = np.asarray(state['param_space'], dtype=object)
        self.facade.param_index = list(state['pending'])
        self.facade.shard_namespace = state.get('shard_namespace')
        self.facade.shard_id = state.get('shard_id')
        self.facade.round_counter = state['round_counter']
        self._generated_count = state['generated_count']
        self._known_values = {key: set(values) for key, values in state.get('known_values', {}).items()}

    def on_domain_changed(self, domain, changed_params):
        new_values = False
        for key in changed_params:
            hashes = {content_hash(value) for value in domain.values_for(key)}
            known = self._known_values.setdefault(key, set())
            new_values = new_values or bool(hashes - known)
            known.update(hashes)
        if not new_values:
            return
        import numpy as np
        facade = self.facade
        known_combos = {content_hash(facade._round_parameters_todict(row)) for row in facade.param_space}
        rows = []
        for values in itertools.product(*(domain.values_for(key) for key in facade.param_keys)):
            combination = facade._round_parameters_todict(values)
            if content_hash(combination) not in known_combos and (facade.boolean_limit is None or facade.boolean_limit(combination)):
                rows.append(values)
        if rows:
            additions = np.empty((len(rows), len(facade.param_keys)), dtype=object)
            for index, values in enumerate(rows):
                for column, value in enumerate(values):
                    additions[index, column] = value
            start = len(facade.param_space)
            facade.param_space = np.vstack([facade.param_space, additions])
            facade.param_index.extend(range(start, start + len(rows)))
