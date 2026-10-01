"""Real-data contracts for native artifacts, public commands and legacy helpers."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from talos.backends import backend_for, normalise_result
from talos.commands.analyze import Analyze
from talos.commands.deploy import Deploy
from talos.commands.evaluate import score
from talos.commands.predict import Predict
from talos.commands.restore import Restore
from talos.utils.validation_split import kfold


@pytest.fixture
def keras_cpu_backend(monkeypatch):
    """Keep scientific reference comparisons portable across available accelerators."""
    monkeypatch.setenv('KERAS_TORCH_DEVICE', 'cpu')
    keras = pytest.importorskip('keras')
    if keras.backend.backend() == 'torch':
        from keras.src.backend.torch.core import device_scope
        with device_scope('cpu'):
            yield keras
    else:
        yield keras


def iris_data():
    x, y = load_iris(return_X_y=True)
    a, b, c, d = train_test_split(x, y, stratify=y, test_size=.2, random_state=17)
    return {'x_train': a, 'x_val': b, 'y_train': c, 'y_val': d}


def test_nondivisible_folds_preserve_real_rows():
    x, y = load_iris(return_X_y=True)
    features, labels = kfold(x[:149], y[:149], folds=7, shuffled=False)
    np.testing.assert_array_equal(np.concatenate(features), x[:149])
    np.testing.assert_array_equal(np.concatenate(labels), y[:149])
    with pytest.raises(ValueError):
        kfold(x, y, folds=151)


def test_legacy_helpers_are_lazy_in_fresh_process():
    code = 'import sys,talos; import talos.utils,talos.autom8,talos.callbacks; assert not any(n in sys.modules for n in ("tensorflow","torch","keras"))'
    subprocess.run([sys.executable, '-c', code], check=True)
    for entrypoint in ('talos.model', 'talos.model.normalizers', 'talos.utils'):
        code = ('import sys; from ' + entrypoint + ' import lr_normalizer; '
                'import talos.model,talos.utils; '
                'assert lr_normalizer is talos.model.lr_normalizer is talos.utils.lr_normalizer; '
                'assert not any(n in sys.modules for n in ("tensorflow","torch","keras"))')
        subprocess.run([sys.executable, '-c', code], check=True)


def test_real_classification_score_encodings():
    x, y = load_breast_cancer(return_X_y=True)
    from sklearn.preprocessing import StandardScaler
    x = StandardScaler().fit_transform(x)
    fitted = LogisticRegression(max_iter=500).fit(x, y)
    probabilities = fitted.predict_proba(x)[:, 1:2]
    expected = f1_score(y, probabilities.ravel() >= .5)
    assert score(y, probabilities, 'binary') == expected
    x, y = load_iris(return_X_y=True)
    fitted = LogisticRegression(max_iter=500).fit(x, y)
    probabilities = fitted.predict_proba(x)
    expected = f1_score(y, probabilities.argmax(axis=-1), average='macro')
    assert score(y, probabilities, 'multi_class') == expected
    assert score(np.eye(3)[y], probabilities, 'multi_label') == expected


def test_commands_and_new_archive(tmp_path):
    x, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=500).fit(x, y)
    scan = SimpleNamespace(data=pd.DataFrame({'val_loss': [float(-model.score(x, y))]}),
                           models=[model], artifacts=[None], params={'C': [1]},
                           details=pd.Series({'experiment_name': 'iris'}), round_history=[], x=x, y=y)
    predicted = Predict(scan).predict(x, 'val_loss', True)
    np.testing.assert_array_equal(predicted, model.predict(x))
    assert Analyze(scan).rounds() == 1
    packaged = Deploy(scan, tmp_path / 'nested' / 'iris', 'val_loss', asc=True)
    restored = Restore(packaged.path)
    np.testing.assert_array_equal(restored.model.predict(x), predicted)
    np.testing.assert_array_equal(restored.x, x[:100])
    assert Analyze(restored).rounds() == 1


def test_generator_and_stopping_presets():
    pytest.importorskip('tensorflow')
    from talos.utils import SequenceGenerator, generator, early_stopper
    x, y = load_iris(return_X_y=True)
    sequence = SequenceGenerator(x=x, y=y, batch_size=64)
    assert [len(sequence[i][0]) for i in range(len(sequence))] == [64, 64, 22]
    stream = generator(x, y, 64)
    assert [len(next(stream)[0]) for _ in range(4)] == [64, 64, 22, 64]
    assert [early_stopper(30, mode=mode).patience for mode in ('lazy', 'moderate', 'strict')] == [10, 3, 2]


def test_stateful_f1_is_batch_partition_invariant(keras_cpu_backend):
    from talos.metrics.keras_metrics import classification_metric
    x, y = load_iris(return_X_y=True)
    probs = LogisticRegression(max_iter=500).fit(x, y).predict_proba(x)
    expected = f1_score(y, probs.argmax(axis=1), average='macro')
    for batch in (1, 7, 150):
        metric = classification_metric()
        for start in range(0, len(y), batch):
            metric.update_state(y[start:start + batch], probs[start:start + batch])
        assert float(metric.result()) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize('backend', ['tensorflow', 'keras', 'torch'])
def test_native_framework_artifact_fresh_process(tmp_path, backend, request):
    if backend == 'keras':
        request.getfixturevalue('keras_cpu_backend')
    pytest.importorskip('torch' if backend == 'torch' else 'tensorflow' if backend == 'tensorflow' else 'keras')
    from talos.experiment.runner import load_sfd
    source = Path(__file__).resolve().parents[1] / 'examples' / 'sfd' / (backend + '_sfd.py')
    if backend == 'torch':
        copied_source = tmp_path / 'user_model.py'
        copied_source.write_text(source.read_text())
        source = copied_source
    sfd = load_sfd(source)
    data = iris_data()
    values = {'neurons': 8, 'learning_rate': .01, 'epochs': 1, 'batch_size': 16}
    normalized = normalise_result(sfd.model(data, values))
    model = normalized['model']
    if backend == 'torch':
        normalized['factory'] = (normalized['factory'], model.talos_config)
    adapter = backend_for(model, backend)
    predictions = adapter.predict(model, data['x_val'])
    descriptor = adapter.save(model, tmp_path / 'model', model_factory=normalized.get('factory'))
    np.save(tmp_path / 'x.npy', data['x_val'])
    (tmp_path / 'descriptor.json').write_text(json.dumps(descriptor))
    scan = SimpleNamespace(data=pd.DataFrame([normalized['metrics']]), models=[model], artifacts=[descriptor],
                           params={'neurons': [8]}, details=pd.Series({'experiment_name': backend}),
                           round_history=[normalized['history']], x=data['x_train'], y=data['y_train'])
    archive = Deploy(scan, tmp_path / 'archive', 'val_loss', asc=True, model_factory=normalized.get('factory'))
    restored = Restore(archive.path)
    np.testing.assert_allclose(backend_for(restored.model, backend).predict(restored.model, data['x_val']), predictions, rtol=1e-5, atol=1e-6)
    code = 'import json,numpy as np; from talos.backends import backend_for; d=json.load(open(%r)); a=backend_for(backend=d["backend"]); m=a.load(d); np.save(%r,a.predict(m,np.load(%r)))' % (str(tmp_path / 'descriptor.json'), str(tmp_path / 'predictions.npy'), str(tmp_path / 'x.npy'))
    code += '; from talos import Restore; r=Restore(%r); np.save(%r,backend_for(r.model).predict(r.model,np.load(%r)))' % (str(archive.path), str(tmp_path / 'archive_predictions.npy'), str(tmp_path / 'x.npy'))
    subprocess.run([sys.executable, '-c', code], check=True, env=os.environ.copy())
    np.testing.assert_allclose(np.load(tmp_path / 'predictions.npy'), predictions, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.load(tmp_path / 'archive_predictions.npy'), predictions, rtol=1e-5, atol=1e-6)
    if backend == 'torch':
        source.unlink()
        archive_only = 'import numpy as np; from talos import Restore; from talos.backends import backend_for; r=Restore(%r); np.save(%r,backend_for(r.model).predict(r.model,np.load(%r)))' % (str(archive.path), str(tmp_path / 'archive_without_source.npy'), str(tmp_path / 'x.npy'))
        subprocess.run([sys.executable, '-c', archive_only], check=True, env=os.environ.copy())
        np.testing.assert_allclose(np.load(tmp_path / 'archive_without_source.npy'), predictions, rtol=1e-5, atol=1e-6)
    content = Path(descriptor['path']).read_bytes()
    Path(descriptor['path']).write_bytes(content + b'tampered')
    with pytest.raises(ValueError, match='checksum mismatch'):
        adapter.load(descriptor)


def test_torch_legacy_return_and_factory_requirement():
    torch = pytest.importorskip('torch')
    from talos.utils import TorchHistory
    class Model(torch.nn.Module, TorchHistory):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(4, 3)
            self.init_history()
        def forward(self, x):
            return self.layer(x)
    net = Model()
    x, y = load_iris(return_X_y=True)
    loss = torch.nn.functional.cross_entropy(net(torch.as_tensor(x, dtype=torch.float32)), torch.as_tensor(y)).item()
    net.append_loss(loss)
    normalized = normalise_result((net, net.parameters()))
    assert normalized['model'] is net
    assert normalized['history']['loss'] == [loss]
    assert not backend_for(net).is_portable(net)


def test_torch_class_factory_archive_survives_source_removal(tmp_path):
    torch = pytest.importorskip('torch')
    import importlib.util
    source = tmp_path / 'user_network.py'
    source.write_text("import torch\nclass Network(torch.nn.Module):\n    def __init__(self, neurons):\n        super().__init__()\n        self.layer = torch.nn.Sequential(torch.nn.Linear(4, neurons), torch.nn.ReLU(), torch.nn.Linear(neurons, 3))\n    def forward(self, x):\n        return self.layer(x)\n")
    module_name = 'talos_test_user_network'
    specification = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    model = module.Network(neurons=7)
    x, y = load_iris(return_X_y=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    loss = torch.nn.functional.cross_entropy(model(torch.as_tensor(x, dtype=torch.float32)), torch.as_tensor(y))
    loss.backward()
    optimizer.step()
    factory = (module.Network, {'neurons': 7})
    normalized = normalise_result({'loss': loss.item(), '_model': model, 'factory': factory})
    adapter = backend_for(model)
    assert adapter.is_portable(model, factory)
    descriptor = adapter.save(model, tmp_path / 'model', factory)
    assert descriptor['config'] == {'neurons': 7}
    assert descriptor['factory_source'] == str(source)
    expected = adapter.predict(model, x)
    np.testing.assert_allclose(adapter.predict(adapter.load(descriptor, model_factory=factory), x), expected)
    scan = SimpleNamespace(data=pd.DataFrame([normalized['metrics']]), models=[model], artifacts=[descriptor],
                           params={'neurons': [7]}, details=pd.Series({'experiment_name': 'torch-class'}),
                           round_history=[], x=x, y=y)
    archive = Deploy(scan, tmp_path / 'archive', 'loss', asc=True)
    source.unlink()
    np.save(tmp_path / 'x.npy', x)
    code = 'import numpy as np; from talos import Restore; from talos.backends import backend_for; r=Restore(%r); np.save(%r,backend_for(r.model).predict(r.model,np.load(%r)))' % (str(archive.path), str(tmp_path / 'restored.npy'), str(tmp_path / 'x.npy'))
    subprocess.run([sys.executable, '-c', code], check=True, env=os.environ.copy())
    np.testing.assert_allclose(np.load(tmp_path / 'restored.npy'), expected)


def test_torch_container_requires_reconstruction_factory(tmp_path):
    torch = pytest.importorskip('torch')
    model = torch.nn.Sequential(torch.nn.Linear(4, 3))
    x, _ = load_iris(return_X_y=True)
    adapter = backend_for(model)
    assert adapter.predict(model, x).shape == (len(x), 3)
    assert not adapter.is_portable(model)
    descriptor = adapter.save(model, tmp_path / 'model')
    assert descriptor['factory_required'] is True
    assert descriptor['factory'] is None
    with pytest.raises(ValueError, match='requires an importable model_factory'):
        adapter.load(descriptor)


def test_recover_model_preserves_loss_parameter_and_numeric_metric(tmp_path):
    from sklearn.metrics import log_loss
    from talos import Scan
    from talos.utils.recover_best_model import recover_best_model
    seen = []
    def train(x_train, y_train, x_val, y_val, params):
        assert set(params) == {'loss', 'C'}
        assert params['loss'] == 'log_loss'
        seen.append(params.copy())
        model = LogisticRegression(C=params['C'], max_iter=500).fit(x_train, y_train)
        loss = log_loss(y_val, model.predict_proba(x_val))
        return SimpleNamespace(history={'loss': [loss], 'val_loss': [loss]}), model
    data = iris_data()
    scan = Scan(data['x_train'], data['y_train'], {'loss': ['log_loss'], 'C': [1.]}, train,
                'recover-iris', x_val=data['x_val'], y_val=data['y_val'],
                experiment_dir=tmp_path / 'run', disable_progress_bar=True)
    assert scan.parameter_columns['loss'] != 'loss'
    assert scan.data[scan.parameter_columns['loss']].iloc[0] == 'log_loss'
    assert isinstance(scan.data['loss'].iloc[0], float)
    recovered, models = recover_best_model(data['x_train'], data['y_train'], data['x_val'], data['y_val'],
                                           scan.run_dir / 'results.csv', train, 'val_loss', n_models=1,
                                           task='multi_class', folds=3)
    assert len(models) == len(recovered) == 1
    assert seen == [{'loss': 'log_loss', 'C': 1.}, {'loss': 'log_loss', 'C': 1.}]


def test_archive_verified_source_bundle_resolves_deleted_helpers(tmp_path):
    pytest.importorskip('torch')
    from zipfile import ZipFile
    from talos import RunResult, run
    from talos.experiment.runner import load_sfd
    helper = tmp_path / 'talos_archive_user_helpers.py'
    helper.write_text("import torch\ndef activation(x):\n    return torch.relu(x)\nclass Network(torch.nn.Module):\n    def __init__(self, width=7):\n        super().__init__()\n        self.hidden = torch.nn.Linear(4, width)\n        self.output = torch.nn.Linear(width, 3)\n    def forward(self, x):\n        return self.output(activation(self.hidden(x)))\ndef make_network(width=7):\n    return Network(width)\n")
    main = tmp_path / 'user_model.py'
    main.write_text("backend = 'torch'\nfrom talos_archive_user_helpers import activation, make_network\ndef params():\n    return {'activation': [activation], 'width': [7]}\ndef prep(data, round_params):\n    return data\ndef model(data, round_params):\n    import torch\n    net = make_network(round_params['width'])\n    optimizer = torch.optim.SGD(net.parameters(), lr=.01)\n    x = torch.as_tensor(data['x_train'], dtype=torch.float32)\n    y = torch.as_tensor(data['y_train'], dtype=torch.long)\n    loss = torch.nn.functional.cross_entropy(net(x), y)\n    loss.backward()\n    optimizer.step()\n    return {'val_loss': loss.item(), '_model': net, 'factory': (make_network, {'width': round_params['width']})}\n")
    data = iris_data()
    result = run(load_sfd(main), data, experiment_dir=tmp_path / 'run', round_limit=1, progress_bar=False)
    main.unlink()
    helper.unlink()
    loaded = RunResult.load(result.run_dir)
    expected = loaded.predict(data['x_val'], metric='val_loss', asc=True)
    packaged = Deploy(loaded, tmp_path / 'portable', 'val_loss', asc=True)
    with ZipFile(packaged.path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        assert 'talos_archive_user_helpers' in manifest['source_bundle']['modules']
        assert all(specification['path'] in archive.namelist() for specification in manifest['source_bundle']['modules'].values()
                   if not specification.get('namespace'))
    np.save(tmp_path / 'x.npy', data['x_val'])
    code = 'import numpy as np; from talos import Restore; from talos.backends import backend_for; r=Restore(%r); assert callable(r.params["activation"][0]); np.save(%r,backend_for(r.model).predict(r.model,np.load(%r)))' % (str(packaged.path), str(tmp_path / 'restored.npy'), str(tmp_path / 'x.npy'))
    subprocess.run([sys.executable, '-c', code], check=True, env=os.environ.copy())
    np.testing.assert_allclose(np.load(tmp_path / 'restored.npy'), expected)
    tampered = tmp_path / 'tampered.zip'
    with ZipFile(packaged.path) as archive, ZipFile(tampered, 'w') as output:
        helper_path = manifest['source_bundle']['modules']['talos_archive_user_helpers']['path']
        for member in archive.infolist():
            content = archive.read(member.filename)
            output.writestr(member, content + b'\n# tampered' if member.filename == helper_path else content)
    with pytest.raises(ValueError, match='source snapshot checksum mismatch'):
        Restore(tampered)


def test_analyze_bivariate_kde_real_iris_density(monkeypatch):
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg', force=True)
    from matplotlib import pyplot as plt
    from matplotlib.contour import QuadContourSet
    x, _ = load_iris(return_X_y=True)
    frame = pd.DataFrame(x[:, :2], columns=['sepal_length', 'sepal_width'])
    analysis = Analyze(frame)
    figure, axes = plt.subplots()
    captured = {}
    contourf = axes.contourf
    def contours(grid_x, grid_y, density, **kwargs):
        captured.update(x=grid_x, y=grid_y, density=density)
        captured['artist'] = contourf(grid_x, grid_y, density, **kwargs)
        return captured['artist']
    monkeypatch.setattr(axes, 'contourf', contours)
    monkeypatch.setattr(analysis, '_axes', lambda: axes)
    result = analysis.plot_kde('sepal_length', 'sepal_width')
    assert result is axes
    assert isinstance(captured['artist'], QuadContourSet)
    density = captured['density']
    assert np.isfinite(density).all()
    assert np.ptp(density) > 0
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    integral = integrate(integrate(density, captured['y'][0], axis=1), captured['x'][:, 0], axis=0)
    assert .95 < integral < 1.005
    assert axes.get_xlabel() == 'sepal_length'
    assert axes.get_ylabel() == 'sepal_width'
    plt.close(figure)


def test_analyze_best_params_aliases_metadata_and_legacy_tables(tmp_path):
    from sklearn.metrics import log_loss
    data = iris_data()
    rows = []
    for index, value in enumerate((.1, 1., 10.)):
        model = LogisticRegression(C=value, max_iter=500).fit(data['x_train'], data['y_train'])
        loss = log_loss(data['y_val'], model.predict_proba(data['x_val']))
        rows.append({'loss': loss, 'val_loss': loss, 'C': value, 'param__loss': 'log_loss',
                     'start': 'start', 'end': 'end', 'duration': 1., 'execution_time': 1.,
                     'round_epochs': 1, '_trial_id': str(index), '_param_hash': str(index), '_warnings': ''})
    frame = pd.DataFrame(rows)
    mapping = {'loss': 'param__loss', 'C': 'C'}
    params = {'loss': ['log_loss'], 'C': [.1, 1., 10.]}
    source = SimpleNamespace(data=frame, params=params, parameter_columns=mapping)
    excluded = ['loss', 'val_loss', 'round_epochs']
    selected = frame.sort_values('loss').head(2)
    expected = np.array([['log_loss', value, index] for index, value in enumerate(selected.C)], dtype=object)
    np.testing.assert_array_equal(Analyze(source).best_params('loss', excluded, n=2, ascending=True), expected)
    np.testing.assert_array_equal(Analyze(source).best_params('loss', ['param__loss'], n=2, ascending=True),
                                  np.column_stack([selected.C, np.arange(2)]))
    frame.to_csv(tmp_path / 'results.csv', index=False)
    (tmp_path / 'metadata.json').write_text(json.dumps({'params': params, 'parameter_columns': mapping}))
    np.testing.assert_array_equal(Analyze(tmp_path / 'results.csv').best_params('loss', excluded, n=2, ascending=True), expected)
    legacy = frame[['loss', 'C', 'execution_time']]
    expected_legacy = legacy.sort_values('loss').drop(columns='loss').head(2).copy()
    expected_legacy['index_num'] = range(2)
    np.testing.assert_array_equal(Analyze(legacy).best_params('loss', [], n=2, ascending=True), expected_legacy.to_numpy())


def binary_iris_data():
    x, y = load_iris(return_X_y=True)
    keep = y < 2
    x, y = x[keep], y[keep]
    a, b, c, d = train_test_split(x, y, stratify=y, test_size=.2, random_state=17)
    return a.astype('float32'), b.astype('float32'), c, d


def test_binary_metrics_compiled_integer_label_fit_matches_sklearn(keras_cpu_backend):
    keras = keras_cpu_backend
    from sklearn.metrics import matthews_corrcoef, precision_score, recall_score
    from talos.metrics.keras_metrics import classification_metric
    x_train, x_val, y_train, y_val = binary_iris_data()
    keras.utils.set_random_seed(17)
    names = ['f1score', 'precision', 'recall', 'matthews']
    model = keras.Sequential([keras.layers.Input((4,)), keras.layers.Dense(1, activation='sigmoid')])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=.01), loss='binary_crossentropy',
                  metrics=[classification_metric(kind) for kind in names])
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2, batch_size=16, verbose=0)
    assert all(len(history.history[name]) == 2 for name in names)
    evaluated = model.evaluate(x_val, y_val, batch_size=7, verbose=0, return_dict=True)
    predicted = model.predict(x_val, verbose=0).reshape(-1) >= .5
    expected = {'f1score': f1_score(y_val, predicted, zero_division=0),
                'precision': precision_score(y_val, predicted, zero_division=0),
                'recall': recall_score(y_val, predicted, zero_division=0),
                'matthews': matthews_corrcoef(y_val, predicted)}
    for name in names:
        assert evaluated[name] == pytest.approx(expected[name], abs=1e-6)


@pytest.mark.parametrize('label_dtype', ['int64', 'float32'])
def test_stateless_fbeta_compiled_repeated_batches_and_scientific_score(label_dtype, keras_cpu_backend):
    keras = keras_cpu_backend
    from sklearn.metrics import fbeta_score
    from talos.metrics.keras_metrics import fbeta
    x_train, x_val, y_train, y_val = binary_iris_data()
    model = keras.Sequential([keras.layers.Input((4,)), keras.layers.Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[fbeta])
    history = model.fit(x_train, y_train.astype(label_dtype), epochs=2, batch_size=7, verbose=0)
    assert len(history.history['fbeta']) == 2
    assert np.isfinite(history.history['fbeta']).all()
    fitted = LogisticRegression(max_iter=500).fit(x_train, y_train)
    probabilities = fitted.predict_proba(x_val)[:, 1:2]
    for beta in (0., .5, 1., 2.):
        expected = fbeta_score(y_val, probabilities.reshape(-1) >= .5, beta=beta, zero_division=0)
        assert float(fbeta(y_val.astype(label_dtype), probabilities, beta)) == pytest.approx(expected, abs=1e-6)
    multiclass = iris_data()
    fitted = LogisticRegression(max_iter=500).fit(multiclass['x_train'], multiclass['y_train'])
    probabilities = fitted.predict_proba(multiclass['x_val'])
    expected = fbeta_score(multiclass['y_val'], probabilities.argmax(axis=1), beta=2, average='macro', zero_division=0)
    assert float(fbeta(multiclass['y_val'], probabilities, beta=2)) == pytest.approx(expected, abs=1e-6)


def test_titanic_callback_accepts_matching_optimizer_parameter():
    pytest.importorskip('tensorflow')
    from talos.templates import models, params
    from talos.model import lr_normalizer
    x_train, x_val, y_train, y_val = binary_iris_data()
    candidates = params.titanic(debug=True)
    values = {name: choices[0] for name, choices in candidates.items()}
    values['epochs'] = 1
    history, model = models.titanic(x_train, y_train, x_val, y_val, values)
    assert len(history.history['loss']) == 1
    assert np.isfinite(history.history['val_loss']).all()
    assert float(model.optimizer.learning_rate.numpy()) == pytest.approx(lr_normalizer(values['lr'], values['optimizer']))


def test_packaged_sfd_templates_expose_lazy_backend_hints():
    from importlib import import_module
    for name, expected in [('keras', 'keras'), ('tf_keras', 'tensorflow'), ('pytorch', 'torch')]:
        template = import_module('talos.sfd.templates.' + name)
        assert template.backend == expected


@pytest.mark.parametrize('label_dtype', ['int64', 'float64'])
def test_continuous_helpers_compiled_diabetes_labels_and_shape_match_sklearn(label_dtype, keras_cpu_backend):
    keras = keras_cpu_backend
    from sklearn.datasets import load_diabetes
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, mean_squared_log_error
    from talos.metrics import keras_metrics
    x, y = load_diabetes(return_X_y=True)
    x, y = x[:40], y[:40].astype(label_dtype)
    names = ['mae', 'mse', 'rmae', 'rmse', 'mape', 'msle', 'rmsle']
    model = keras.Sequential([keras.layers.Input((10,)), keras.layers.Dense(1, activation='softplus')])
    model.compile(optimizer='adam', loss='mse', metrics=[getattr(keras_metrics, name) for name in names])
    trained = model.train_on_batch(x, y, return_dict=True)
    assert all(np.isfinite(trained[name]) for name in names)
    evaluated = model.evaluate(x, y, batch_size=7, verbose=0, return_dict=True)
    predictions = model.predict(x, verbose=0).reshape(-1)
    expected = {'mae': mean_absolute_error(y, predictions), 'mse': mean_squared_error(y, predictions),
                'mape': 100 * mean_absolute_percentage_error(y, predictions),
                'msle': mean_squared_log_error(y, predictions)}
    for name, value in expected.items():
        assert evaluated[name] == pytest.approx(value, rel=1e-5, abs=1e-5)
    # Root helpers retain their per-example target-axis reduction before Keras aggregation.
    expected_roots = {'rmae': np.sqrt(np.abs(y - predictions)), 'rmse': np.abs(y - predictions),
                      'rmsle': np.abs(np.log1p(y) - np.log1p(predictions))}
    for name, values in expected_roots.items():
        computed = getattr(keras_metrics, name)(y, predictions[:, None])
        converted = keras.ops.convert_to_numpy(computed) if hasattr(keras, 'ops') else keras.backend.get_value(computed)
        np.testing.assert_allclose(converted, values, rtol=1e-5, atol=1e-5)
        assert evaluated[name] == pytest.approx(np.mean(values), rel=1e-5, abs=1e-5)


@pytest.mark.parametrize('framework', ['numpy', 'tensorflow', 'torch'])
def test_normalise_history_validates_all_points_and_framework_scalars(framework):
    if framework == 'tensorflow':
        tf = pytest.importorskip('tensorflow')
        scalar = lambda value: tf.constant(value, dtype=tf.float64)
    elif framework == 'torch':
        torch = pytest.importorskip('torch')
        scalar = lambda value: torch.tensor(value, dtype=torch.float64)
    else:
        scalar = np.float64
    from sklearn.metrics import log_loss
    data = iris_data()
    model = LogisticRegression(max_iter=500).fit(data['x_train'], data['y_train'])
    training = log_loss(data['y_train'], model.predict_proba(data['x_train']))
    validation = log_loss(data['y_val'], model.predict_proba(data['x_val']))
    expected = [training, validation]
    result = normalise_result({'_history': {'loss': [scalar(value) for value in expected]},
                               'metrics': {'val_loss': scalar(validation)}})
    assert result['history']['loss'] == pytest.approx(expected)
    assert all(isinstance(point, float) for point in result['history']['loss'])
    assert result['metrics']['loss'] == pytest.approx(validation)
    assert result['metrics']['val_loss'] == pytest.approx(validation)
    if framework == 'torch':
        low_precision = torch.tensor(validation, dtype=torch.bfloat16)
        normalized = normalise_result({'_history': {'loss': [low_precision]}})
        assert normalized['history']['loss'] == [low_precision.item()]
    for output in ({'_history': {'loss': ['bad', scalar(validation)]}},
                   (SimpleNamespace(history={'loss': ['bad', scalar(validation)]}), model)):
        with pytest.raises(TypeError, match=r'History loss\[0\] must be a real numeric scalar'):
            normalise_result(output)
    with pytest.raises(TypeError, match='must contain a sequence'):
        normalise_result({'_history': {'loss': scalar(validation)}})
