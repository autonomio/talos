import logging
import math
import random
import re
import time
import warnings
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any

from typing_extensions import override

from talos.yaml.schema import COMPLEXITY_HIGH_MAX
from talos.yaml.schema import COMPLEXITY_LOW_MAX
from talos.yaml.schema import COMPLEXITY_MEDIUM_MAX

if TYPE_CHECKING:
    from talos.yaml.compiler import CompiledSFD


_COMPLEXITY_THRESHOLDS = [
    (COMPLEXITY_LOW_MAX, 'low'),
    (COMPLEXITY_MEDIUM_MAX, 'medium'),
    (COMPLEXITY_HIGH_MAX, 'high'),
]


@dataclass
class ProfileResult:

    '''Summary of a YAML experiment profile run.'''

    total_permutations: int
    param_cardinalities: dict[str, int]
    complexity_rating: str
    sample_permutations_attempted: int = 0
    sample_permutations_completed: int = 0
    sample_time_seconds_per_permutation: float | None = None
    warnings: list[str] = field(default_factory=list[str])
    errors: list[str] = field(default_factory=list[str])
    data_quality_warnings: list[str] = field(default_factory=list[str])


_MANIFEST_CORE_LOGGER = 'talos.experiment.runner'


class _LogCapture(logging.Handler):

    def __init__(self) -> None:

        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    @override
    def emit(self, record: logging.LogRecord) -> None:

        self.records.append(record.getMessage())


def _complexity_rating(total: int) -> str:

    for threshold, label in _COMPLEXITY_THRESHOLDS:
        if total <= threshold:
            return label
    return 'extreme'



def make_covering_array(params: dict[str, list[Any]], seed: int = 42) -> list[dict[str, Any]]:

    '''
    Build a randomised strength-1 covering array for the given parameter space.

    Every value of every parameter appears in at least one permutation.
    N = max cardinality across all params. Each parameter's column is filled
    with round-robin values then shuffled independently, removing systematic
    correlation between parameter assignments.

    Args:
        params (dict): Parameter name to list of values
        seed (int): Random seed for reproducibility

    Returns:
        list[dict]: N permutations, each a dict of param name to sampled value

    '''

    if not params:
        return []

    n = max(len(v) for v in params.values())
    rng = random.Random(seed)
    columns: dict[str, list[Any]] = {}

    for key, values in params.items():
        if not values:
            raise ValueError(f"Parameter '{key}' has an empty values list")
        column = [values[i % len(values)] for i in range(n)]
        rng.shuffle(column)
        columns[key] = column

    return [{key: columns[key][i] for key in params} for i in range(n)]


def _classify_error(exc: Exception) -> str:

    msg = str(exc)
    lower = msg.lower()
    if 'sample' in lower and ('0 ' in lower or 'empty' in lower):
        return f'Data too small for profiling — supply more observations through the caller SFD. ({msg})'
    if 'class' in lower and ('one' in lower or 'least' in lower):
        return f'Not enough class diversity in test data — supply more caller observations. ({msg})'
    if 'nan' in lower or re.search(r'\binf\b', lower):
        return f'Test data contains NaN or Inf values. ({msg})'
    return f'{type(exc).__name__}: {msg}'


def profile(compiled_sfd, data=None, *, runtime=True) -> ProfileResult:
    params = compiled_sfd.params()
    cards = {key: len(values) for key, values in params.items()}
    total = math.prod(cards.values())
    result = ProfileResult(total, cards, _complexity_rating(total))
    if not runtime:
        return result
    permutations = make_covering_array(params)
    result.sample_permutations_attempted = len(permutations)
    elapsed = []
    quality = set()
    for round_params in permutations:
        start = time.perf_counter()
        try:
            prepared = compiled_sfd.prep(data, round_params)
            quality.update(_quality_warnings(prepared))
            output = compiled_sfd.model(prepared, round_params)
            elapsed.append(time.perf_counter() - start)
            from talos.backends import normalise_result
            normalized = normalise_result(output)
            model = normalized['model']
            if model is not None:
                from talos.backends import backend_for
                backend_for(model).cleanup()
        except Exception as exc:
            result.errors.append(_classify_error(exc))
    result.sample_permutations_completed = len(elapsed)
    result.data_quality_warnings = sorted(quality)
    if elapsed:
        result.sample_time_seconds_per_permutation = sum(elapsed) / len(elapsed)
    if result.sample_permutations_completed != result.sample_permutations_attempted:
        result.warnings.append('Timing includes successful sampled permutations only.')
    return result


def _quality_warnings(value, label='prepared', seen=None):
    """Inspect already prepared caller arrays/tables without reading input data."""
    import numpy as np
    seen = set() if seen is None else seen
    if id(value) in seen:
        return []
    seen.add(id(value))
    if isinstance(value, dict):
        return [warning for key, part in value.items() for warning in _quality_warnings(part, f'{label}.{key}', seen)]
    if isinstance(value, (list, tuple)):
        return [warning for index, part in enumerate(value) for warning in _quality_warnings(part, f'{label}[{index}]', seen)]
    if hasattr(value, 'to_numpy'):
        value = value.to_numpy()
    if not isinstance(value, np.ndarray) or not np.issubdtype(value.dtype, np.number):
        return []
    warnings = []
    missing, infinite = int(np.isnan(value).sum()), int(np.isinf(value).sum())
    if missing:
        warnings.append(f'{label}: {missing} NaN values in caller-prepared data')
    if infinite:
        warnings.append(f'{label}: {infinite} Inf values in caller-prepared data')
    return warnings
