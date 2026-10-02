"""Train each shipped caller-data template on bundled scientific datasets."""
import importlib

import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from talos.backends import backend_for, normalise_result


@pytest.fixture(params=['keras', 'tf_keras', 'pytorch'])
def packaged_template(request, monkeypatch):
    monkeypatch.setenv('KERAS_TORCH_DEVICE', 'cpu')
    name = request.param
    module = importlib.import_module('talos.sfd.templates.' + name)
    if name == 'pytorch':
        torch = pytest.importorskip('torch')
        torch.manual_seed(17)
        yield module
    else:
        keras = pytest.importorskip('tensorflow').keras if name == 'tf_keras' else pytest.importorskip('keras')
        keras.utils.set_random_seed(17)
        if keras.backend.backend() == 'torch':
            from keras.src.backend.torch.core import device_scope
            with device_scope('cpu'):
                yield module
        else:
            yield module
        keras.backend.clear_session()


def prepared_observations(task):
    if task == 'regression':
        x, y = load_diabetes(return_X_y=True)
        y = y.astype('float32') / 200
        stratify = None
    else:
        x, y = load_iris(return_X_y=True)
        if task == 'binary':
            x, y = x[y < 2], y[y < 2]
        stratify = y
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=.2, stratify=stratify, random_state=17)
    scaler = StandardScaler().fit(x_train)
    if task == 'one_hot':
        y_train, y_val = np.eye(3, dtype='float32')[y_train], np.eye(3, dtype='float32')[y_val]
    return {'x_train': scaler.transform(x_train).astype('float32'),
            'x_val': scaler.transform(x_val).astype('float32'),
            'y_train': y_train, 'y_val': y_val,
            'task': 'multiclass' if task == 'one_hot' else task}


@pytest.mark.parametrize('task', ['regression', 'binary', 'multiclass', 'one_hot'])
def test_packaged_template_learns_and_restores_caller_task(packaged_template, task, tmp_path):
    template = packaged_template
    prepared = prepared_observations(task)
    assert template.prep(prepared) is prepared
    params = {'units': 16, 'epochs': 6, 'batch_size': len(prepared['x_train']), 'learning_rate': .02}
    output = normalise_result(template.model(prepared, params), backend=template.backend)
    loss = output['history']['loss']
    assert len(loss) == params['epochs']
    assert np.isfinite(loss).all()
    assert loss[-1] < loss[0]
    adapter = backend_for(output['model'], template.backend)
    prediction = adapter.predict(output['model'], prepared['x_val'])
    outputs = 3 if task in ('multiclass', 'one_hot') else 1
    assert prediction.shape == (len(prepared['x_val']), outputs)
    assert np.isfinite(prediction).all()
    if task != 'regression' and template.backend != 'torch':
        assert np.logical_and(prediction >= 0, prediction <= 1).all()
        if outputs == 3:
            np.testing.assert_allclose(prediction.sum(axis=1), 1, rtol=1e-5)
    descriptor = adapter.save(output['model'], tmp_path / 'trained')
    restored = adapter.load(descriptor)
    np.testing.assert_allclose(adapter.predict(restored, prepared['x_val']), prediction,
                               rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('name', ['keras', 'tf_keras', 'pytorch'])
def test_packaged_templates_require_caller_data_and_expose_parameter_sweeps(name):
    template = importlib.import_module('talos.sfd.templates.' + name)
    with pytest.raises(ValueError, match='Supply caller data'):
        template.prep(None)
    grid = template.params()
    assert grid['units'] == [16, 32]
    assert all(isinstance(values, list) and values for values in grid.values())
