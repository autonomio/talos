"""Resolve parameter indicators and candidate groups for legacy reducers."""

import numpy as np
import pandas as pd

from talos.experiment.serialization import content_hash


def trial_frame(scan):
    result = getattr(scan, 'result', None)
    if isinstance(result, pd.DataFrame):
        data = result.copy()
    elif isinstance(result, list) and result:
        if isinstance(result[0], dict):
            data = pd.DataFrame(result)
        else:
            data = pd.DataFrame(result[1:], columns=result[0])
    elif isinstance(getattr(scan, 'data', None), pd.DataFrame):
        data = scan.data.copy()
    else:
        data = pd.read_csv(scan._experiment_log)
    return data.tail(scan.reduction_window)


def parameter_indicators(scan):
    data = trial_frame(scan)
    metric = scan.reduction_metric
    if metric not in data:
        raise ValueError(f'Reduction metric {metric!r} is absent from trial results.')
    target = pd.to_numeric(data[metric], errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(target)
    target = target[valid]
    columns, candidates = [], []
    parameter_columns = getattr(scan, 'parameter_columns', {})
    for label in scan._param_dict_keys:
        result_column = parameter_columns.get(label, label)
        if result_column not in data:
            continue
        values = data[result_column].to_numpy(dtype=object)[valid]
        identifiers = [content_hash(value) for value in values]
        seen = set()
        for value, identifier in zip(values, identifiers):
            if identifier not in seen:
                seen.add(identifier)
                indicator = np.array([key == identifier for key in identifiers], dtype=float)
                if len(set(indicator)) > 1:
                    columns.append(indicator)
                    candidates.append((label, value))
    matrix = np.column_stack(columns) if columns else np.empty((len(target), 0))
    return matrix, target, candidates


def cols_to_multilabel(scan):
    matrix, target, candidates = parameter_indicators(scan)
    data = pd.DataFrame(matrix, columns=range(len(candidates)))
    data[scan.reduction_metric] = target
    data.attrs['parameter_values'] = candidates
    return data
