import math
import random
from numbers import Integral


def sample_reducer(limit, max_value, random_method, seed=None):
    """Sample unique legacy Cartesian row indexes with the selected method."""
    n = int(max_value * limit) if isinstance(limit, float) else int(limit)
    max_value = int(max_value)
    if n < 1:
        from talos.utils.exceptions import TalosDataError
        raise TalosDataError('Limiters lead to < 1 permutations.')
    n = min(n, max_value)
    if n == 0:
        return []
    if random_method == 'uniform_mersenne':
        return random.Random(seed).sample(range(max_value), n)
    if random_method == 'uniform_crypto':
        return random.SystemRandom().sample(range(max_value), n)
    if random_method in ('quantum', 'ambience'):
        try:
            from chances import Randomizer
        except ImportError as error:
            raise ImportError(f'{random_method} sampling requires the optional chances package.') from error
        selected = []
        seen = set()
        for _ in range(32):
            candidates = getattr(Randomizer(max_value, n - len(selected)), random_method)()
            for value in candidates:
                if isinstance(value, Integral) and not isinstance(value, bool) and 0 <= value < max_value:
                    value = int(value)
                    if value not in seen:
                        selected.append(value)
                        seen.add(value)
                        if len(selected) == n:
                            return selected
        from talos.utils.exceptions import TalosDataError
        raise TalosDataError(f'{random_method} service did not supply {n} unique indexes in [0, {max_value}).')
    if random_method == 'korobov_matrix':
        generator = max(1, int(max_value * .6180339887498949))
        while math.gcd(generator, max_value) != 1:
            generator += 1
        return [(index * generator) % max_value for index in range(n)]
    import numpy as np
    if random_method == 'latin_improved':
        # Legacy improved LHD: greedily maximize distance from chosen bins.
        rng = np.random.default_rng(seed)
        available = list(range(max_value))
        chosen = [available.pop(int(rng.integers(len(available))))]
        while len(chosen) < n:
            candidates = rng.choice(available, size=min(100, len(available)), replace=False)
            distances = np.abs(np.asarray(chosen)[:, None] - candidates[None, :])
            nearest = np.minimum(distances, max_value - distances).min(axis=0)
            value = int(candidates[int(np.argmax(nearest))])
            chosen.append(value)
            available.remove(value)
        return chosen
    from scipy.stats import qmc
    if random_method == 'sobol':
        sample = qmc.Sobol(1, scramble=False, seed=seed).random_base2(math.ceil(math.log2(max_value)))[:max_value, 0]
    elif random_method == 'halton':
        sample = qmc.Halton(1, scramble=False, seed=seed).random(max_value)[:, 0]
    elif random_method in ('latin_matrix', 'latin_sudoku'):
        # Talos' Sudoku adapter uses one box: its first axis is ordinary LHS.
        sample = qmc.LatinHypercube(1, seed=seed).random(max_value)[:, 0]
    else:
        raise ValueError(f'Unknown random_method: {random_method!r}')
    return np.argsort(sample, kind='stable')[:n].tolist()
