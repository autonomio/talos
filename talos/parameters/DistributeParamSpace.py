"""Partition parameter occurrences into stable distributed scan shards."""

import random

import numpy as np

from talos.experiment.serialization import content_hash

from .ParamSpace import ParamSpace


class DistributeParamSpace:
    def __init__(self, params, param_keys=None, random_method='uniform_mersenne',
                 fraction_limit=None, round_limit=None, time_limit=None,
                 boolean_limit=None, machines=2, seed=None):
        if not isinstance(machines, int) or machines < 1:
            raise ValueError('machines must be a positive integer.')
        self._params = ParamSpace(params, param_keys, random_method, fraction_limit, round_limit, time_limit, boolean_limit, seed)
        self.machines = machines
        self.param_spaces = self._split_param_space()

    def _split_param_space(self):
        indices = list(self._params.param_index)
        random.Random(self._params.seed).shuffle(indices)
        out = {}
        for worker, shard in enumerate(np.array_split(indices, self.machines)):
            facade = ParamSpace(self._params.params, self._params.param_keys,
                                time_limit=self._params.time_limit, seed=self._params.seed)
            facade.param_space = self._params.param_space[shard.astype(int)].copy()
            facade.param_index = list(range(len(shard)))
            facade.dimensions = len(shard)
            facade.shard_id = worker
            facade.shard_namespace = content_hash({
                "params": self._params.params, "machines": self.machines,
                "worker": worker, "original_row_indexes": shard.astype(int).tolist()})
            out[worker] = facade
        return out
