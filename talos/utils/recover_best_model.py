from pathlib import Path
import numpy as np
import pandas as pd
from talos.backends import backend_for, normalise_result
from talos.commands.evaluate import score
from talos.utils.validation_split import kfold


def recover_best_model(x_train, y_train, x_val, y_val, experiment_log, input_model,
                       metric, multi_input=False, x_cross=None, y_cross=None,
                       n_models=5, task='multi_label', asc=None, folds=5, param_resolver=None):
    df = pd.read_csv(experiment_log)
    parameter_rows = {}
    parameter_columns = {}
    parameter_keys = None
    run_dir = Path(experiment_log).resolve().parent
    if (run_dir / 'metadata.json').is_file() and (run_dir / 'round_data.jsonl').is_file():
        from talos.experiment.runner import RunResult
        from talos.experiment.artifacts import read_rounds
        logged = RunResult.load(run_dir)
        parameter_columns = getattr(logged, 'parameter_columns', {})
        parameter_keys = list(logged.params)
        for index, record in enumerate(read_rounds(run_dir / 'round_data.jsonl', len(logged.data))):
            if 'params' in record:
                parameter_rows[index] = dict(record['params'])
            else:
                row = logged.data.iloc[index]
                parameter_rows[index] = {key: row[parameter_columns.get(key, key)] for key in parameter_keys}
    if asc is None:
        asc = any(name in metric.lower() for name in ('loss', 'error', 'mae', 'mse'))
    candidates = df.sort_values(metric, ascending=asc).head(n_models).copy()
    if (x_cross is None) != (y_cross is None):
        raise ValueError('Provide both x_cross and y_cross.')
    if x_cross is None:
        x_cross, y_cross = x_val, y_val
    results, models = [], []
    for model_id, row in candidates.iterrows():
        if model_id in parameter_rows:
            params = dict(parameter_rows[model_id])
        elif parameter_keys is not None:
            params = {key: row[parameter_columns.get(key, key)] for key in parameter_keys}
        else:
            params = row.drop(metric).to_dict()
        if param_resolver:
            params = param_resolver(params)
        result = normalise_result(input_model(x_train, y_train, x_val, y_val, params))
        model = result['model']
        adapter = backend_for(model, result['backend'])
        kx, ky = kfold(x_cross, y_cross, folds, True, multi_input)
        values = [score(labels, adapter.predict(model, features), task) for features, labels in zip(kx, ky)]
        results.append(float(np.mean(values)))
        models.append(model)
    name = 'crossval_mean_mae' if task in ('continuous', 'regression') else 'crossval_mean_f1score'
    candidates[name] = results
    return candidates, models
