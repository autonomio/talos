"""Caller SFD/CLI persistence contracts; parameter-only probes and real Iris data."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from talos.cli.main import cli
from talos.yaml.compiler import CompiledSFD
from talos.yaml.config import round_trip_yaml
from talos.yaml.store import canonical_manifest_id
from talos.yaml.validator import validate


def manifest(module='sfd.py', mode='development'):
    return {'schema_version': '1.0', 'metadata': {'name': 'contract', 'mode': mode},
            'sfd': {'module': module, 'objective': {'metric': 'score', 'direction': 'max'}},
            'uel': {'search_strategy': {'type': 'grid'}, 'round_limit': 2, 'seed': 42}}


def write_manifest(path, document):
    from io import StringIO
    stream = StringIO()
    round_trip_yaml().dump(document, stream)
    path.write_text(stream.getvalue())


@pytest.fixture
def project(tmp_path):
    (tmp_path / 'talos.toml').write_text('[store]\nbackup_remote = ""\n')
    (tmp_path / 'sfd.py').write_text(
        "def params(): return {'rate': [1, 2]}\n"
        "def prep(context, round_params): return context\n"
        "def model(context, round_params): return {'score': round_params['rate'], 'history': {'score': [round_params['rate']]}}\n")
    path = tmp_path / 'experiment.yaml'
    write_manifest(path, manifest())
    return tmp_path, path


def test_all_commands_and_local_framework_templates(tmp_path, monkeypatch):
    runner = CliRunner()
    assert set(cli.commands) == {'new', 'backup', 'init', 'list-templates', 'validate', 'profile',
                                 'commit', 'ls', 'run', 'reindex', 'fork', 'lineage'}
    monkeypatch.chdir(tmp_path)
    made = runner.invoke(cli, ['new', 'caller-project'])
    assert made.exit_code == 0, made.output
    monkeypatch.chdir(tmp_path / 'caller-project')
    for name in ['keras', 'tf_keras', 'pytorch']:
        initialized = runner.invoke(cli, ['init', name, '--template', name])
        assert initialized.exit_code == 0, initialized.output
        assert (Path('manifests') / f'{name}_sfd.py').is_file()
        checked = runner.invoke(cli, ['run', '--dry-run', str(Path('manifests') / f'{name}.yaml')])
        assert checked.exit_code == 0, checked.output


def test_parameter_only_execute_resume_and_manifest_metadata(project):
    root, path = project
    compiled = CompiledSFD(manifest(), source_path=path)
    run_dir = root / 'run'
    result = compiled.execute(experiment_dir=run_dir, progress_bar=False)
    assert result.data.score.tolist() == [1, 2]
    reference = json.loads((run_dir / 'metadata.json').read_text())['yaml_reference']
    assert reference['manifest_id'] == canonical_manifest_id(manifest())
    assert reference['source_path'] == str(path)
    resumed = CliRunner().invoke(cli, ['run', '--resume', str(run_dir), '--no-progress-bar'])
    assert resumed.exit_code == 0, resumed.output
    assert len((run_dir / 'round_data.jsonl').read_text().splitlines()) == 2


def test_cli_profile_run_and_store_lineage(project, monkeypatch):
    root, path = project
    monkeypatch.chdir(root)
    runner = CliRunner()
    for command in [['validate', str(path)], ['profile', str(path)], ['run', '--dry-run', str(path)],
                    ['run', '--no-progress-bar', str(path)]]:
        result = runner.invoke(cli, command)
        assert result.exit_code == 0, result.output
    outputs = list(root.glob('results/dev/*/results.csv'))
    assert len(outputs) == 1
    subprocess.run(['git', 'init', '-q', str(root)], check=True)
    subprocess.run(['git', '-C', str(root), 'config', 'user.email', 'test@example.invalid'], check=True)
    subprocess.run(['git', '-C', str(root), 'config', 'user.name', 'Contract test'], check=True)
    write_manifest(path, manifest(mode='production'))
    result = runner.invoke(cli, ['commit', str(path)])
    assert result.exit_code == 0, result.output
    identifier = canonical_manifest_id(manifest(mode='production'))
    for command in [['ls'], ['reindex'], ['lineage', identifier], ['fork', identifier, 'next'],
                    ['run', '--dry-run', 'manifest://' + identifier]]:
        result = runner.invoke(cli, command)
        assert result.exit_code == 0, result.output
    assert (root / 'manifests' / 'next.yaml').exists()


def test_financial_schema_rejected_and_explicit_callable_values(project):
    _, path = project
    document = manifest()
    document['sfd']['manifest'] = {'data_source': 'anything'}
    assert not validate(document).valid
    document = manifest()
    document['sfd']['params'] = {'rate': [{'callable': 'builtins:abs'}]}
    assert CompiledSFD(document, source_path=path).params()['rate'] == [abs]


def test_sensor_and_cohort_restore_real_iris(tmp_path):
    from talos.cohort import Cohort
    from talos.inference import Sensor
    source = tmp_path / 'iris_sfd.py'
    source.write_text(
        "from sklearn.datasets import load_iris\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "def params(): return {'c': [0.5, 1.0]}\n"
        "def prep(context, round_params): return load_iris(return_X_y=True)\n"
        "def model(prepared, round_params):\n"
        "    x, y = prepared\n"
        "    model = LogisticRegression(C=round_params['c'], max_iter=300).fit(x, y)\n"
        "    return {'score': model.score(x, y), 'model': model}\n")
    document = manifest(source.name)
    compiled = CompiledSFD(document, source_path=tmp_path / 'iris.yaml')
    result = compiled.execute(experiment_dir=tmp_path / 'trained', progress_bar=False)
    from sklearn.datasets import load_iris
    x, y = load_iris(return_X_y=True)
    sensor = Sensor(run_dir=result.run_dir, metric='score')
    predictions = sensor.predict(x)
    assert predictions.shape == y.shape
    assert (predictions == y).mean() > 0.9
    ensemble = Cohort(result=result, aggregation='vote', task='multiclass')
    assert ensemble.predict(x).shape == y.shape
    top = Cohort(result=result, selector='top_n', selector_params={'column': 'score', 'n': 1})
    assert len(top.permutation_ids) == 1


def test_general_scalers_work_without_financial_column_names():
    import polars as pl
    from sklearn.datasets import load_iris

    from talos.scalers import LinearScaler, LogRegScaler, RankGaussScaler
    from talos.scalers.linear_scaler import inverse_transform
    frame = pl.DataFrame(load_iris().data, schema=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    for scaler_class in [LinearScaler, LogRegScaler]:
        scaler = scaler_class(frame)
        transformed = scaler.transform(frame)
        np.testing.assert_allclose(transformed.to_numpy().mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(inverse_transform(transformed, scaler).to_numpy(), frame.to_numpy())
    assert np.isfinite(RankGaussScaler(frame).transform(frame).to_numpy()).all()


def test_resume_rejects_changed_caller_source(project):
    root, path = project
    compiled = CompiledSFD(manifest(), source_path=path)
    run_dir = root / 'run'
    compiled.execute(experiment_dir=run_dir, progress_bar=False)
    source = root / 'sfd.py'
    source.write_text(source.read_text().replace("round_params['rate'], 'history'", "round_params['rate'] * 10, 'history'"))
    resumed = CliRunner().invoke(cli, ['run', '--resume', str(run_dir), '--no-progress-bar'])
    assert resumed.exit_code == 1
    assert any(reason in resumed.output.lower() for reason in ['hash', 'identity', 'content', 'source changed'])


def test_local_git_backup_and_restore(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    remote = tmp_path / 'backup.git'
    subprocess.run(['git', 'init', '--bare', '-q', str(remote)], check=True)
    runner = CliRunner()
    created = runner.invoke(cli, ['new', 'source', '--backup-remote', str(remote)])
    assert created.exit_code == 0, created.output
    project = tmp_path / 'source'
    subprocess.run(['git', '-C', str(project), 'config', 'user.email', 'test@example.invalid'], check=True)
    subprocess.run(['git', '-C', str(project), 'config', 'user.name', 'Contract test'], check=True)
    monkeypatch.chdir(project)
    (project / 'caller.py').write_text('def prep(context): return context\n')
    backed_up = runner.invoke(cli, ['backup'])
    assert backed_up.exit_code == 0, backed_up.output
    subprocess.run(['git', '--git-dir', str(remote), 'symbolic-ref', 'HEAD', 'refs/heads/main'], check=True)
    monkeypatch.chdir(tmp_path)
    restored = runner.invoke(cli, ['new', 'restored', '--from', str(remote)])
    assert restored.exit_code == 0, restored.output
    assert (tmp_path / 'restored' / 'caller.py').read_text() == (project / 'caller.py').read_text()
    assert (tmp_path / 'restored' / '.git').exists()


def test_generic_log_multiclass_and_regression_real_iris(tmp_path):
    from sklearn.datasets import load_iris

    from talos.log import Log
    x, y = load_iris(return_X_y=True)
    artifact = tmp_path / 'results.csv'
    artifact.write_text('score\n1.0\n')
    classification = Log(file_path=artifact, predictions=np.eye(3)[y], targets=y)
    assert classification.permutation_prediction_performance()['accuracy'] == 1.0
    assert classification.permutation_confusion_metrics()['confusion_matrix'].shape == (3, 3)
    regression = Log(file_path=artifact, predictions=x[:, 0], targets=x[:, 0])
    assert regression.permutation_prediction_performance(task='regression')['mse'] == 0


def test_pytorch_project_template_fresh_process_artifact_restore(tmp_path):
    import importlib.util
    if importlib.util.find_spec('torch') is None:
        pytest.skip('PyTorch extra is not installed')
    import sys
    source = tmp_path / 'pytorch_sfd.py'
    template = Path(__file__).resolve().parents[1] / 'talos' / 'sfd' / 'templates' / 'pytorch.py'
    source.write_text(template.read_text() + '\n\ndef prep(context=None, round_params=None):\n'
                      '    from sklearn.datasets import load_iris\n'
                      '    x, y = load_iris(return_X_y=True)\n'
                      "    return {'x_train': x, 'y_train': y, 'task': 'multiclass'}\n")
    document = manifest(source.name)
    document['sfd'].update({'backend': 'torch', 'objective': 'loss',
                            'params': {'units': [4], 'epochs': [1], 'learning_rate': [0.001]}})
    document['uel']['round_limit'] = 1
    compiled = CompiledSFD(document, source_path=tmp_path / 'torch.yaml')
    result = compiled.execute(experiment_dir=tmp_path / 'torch-trained', progress_bar=False)
    from sklearn.datasets import load_iris
    x, _ = load_iris(return_X_y=True)
    expected = result.predict(x)
    output = tmp_path / 'restored.json'
    script = ('import json; from sklearn.datasets import load_iris; '
              'from talos.inference import Sensor; '
              f'sensor=Sensor(run_dir={str(result.run_dir)!r}); '
              'x,_=load_iris(return_X_y=True); '
              f'open({str(output)!r},"w").write(json.dumps(sensor.predict(x).tolist()))')
    restored = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert restored.returncode == 0, restored.stderr
    np.testing.assert_allclose(json.loads(output.read_text()), expected, rtol=1e-6)


def test_held_out_iris_calibration_multiclass_and_binary():
    from types import SimpleNamespace

    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from talos.calibration import (
        apply_calibrated_predict,
        grid_threshold_optimizer,
        sklearn_probability_calibrator,
    )
    x, y = load_iris(return_X_y=True)
    train_x, held_x, train_y, held_y = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)
    val_x, test_x, val_y, test_y = train_test_split(held_x, held_y, test_size=0.5, random_state=43, stratify=held_y)
    classifier = LogisticRegression(max_iter=300).fit(train_x, train_y)
    config = SimpleNamespace(calibration_func=sklearn_probability_calibrator,
                             calibration_params={'method': 'sigmoid'}, threshold_func=None, threshold_params={})
    output = apply_calibrated_predict(classifier, config, {'x_val': val_x, 'y_val': val_y, 'x_test': test_x})
    assert output['_probs'].shape == (len(test_y), 3)
    np.testing.assert_allclose(output['_probs'].sum(axis=1), 1)
    assert output['optimal_threshold'] is None
    assert (output['_preds'] == test_y).mean() > 0.8
    binary_model = LogisticRegression(max_iter=300).fit(train_x, (train_y == 2).astype(int))
    config.threshold_func = grid_threshold_optimizer
    output = apply_calibrated_predict(binary_model, config,
                                      {'x_val': val_x, 'y_val': (val_y == 2).astype(int), 'x_test': test_x})
    assert output['_probs'].shape == (len(test_y),)
    assert 0 <= output['optimal_threshold'] <= 1
    assert set(output['_preds']) <= {0, 1}


def test_preparation_preserves_actual_iris_rows_with_and_without_seed():
    import polars as pl
    from sklearn.datasets import load_iris

    from talos.preparation import split_data_to_prep_output, split_random, split_sequential
    iris = load_iris()
    frame = pl.DataFrame(iris.data, schema=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    frame = frame.with_columns(pl.Series('species', iris.target)).with_row_index('_row')
    for splitter in [lambda: split_sequential(frame, [3, 1, 1]),
                     lambda: split_random(frame, [3, 1, 1]),
                     lambda: split_random(frame, [3, 1, 1], seed=42)]:
        splits = splitter()
        assert [len(split) for split in splits] == [90, 30, 30]
        rows = [row for split in splits for row in split['_row']]
        assert sorted(rows) == list(range(len(frame)))
        output = split_data_to_prep_output(splits, cols=frame.columns[1:], target_cols='species', as_numpy=True)
        assert output['x_train'].shape == (90, 4)
        assert output['y_train'].shape == (90,)
        assert '_alignment' not in output
    with pytest.raises(ValueError):
        split_random(frame, [0, 0])


def test_trainer_retrain_validation_and_generic_prediction_diagnostics(tmp_path):
    from sklearn.datasets import load_iris

    from talos.inference import Trainer
    from talos.log import Log
    source = tmp_path / 'iris_retrain.py'
    source.write_text(
        "from sklearn.datasets import load_iris\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "def params(): return {'c': [0.5, 1.0]}\n"
        "def prep(context, round_params): return load_iris(return_X_y=True)\n"
        "def model(prepared, round_params):\n"
        "    x, y = prepared\n"
        "    classifier = LogisticRegression(C=round_params['c'], max_iter=300).fit(x, y)\n"
        "    return {'score': classifier.score(x, y), 'model': classifier}\n")
    compiled = CompiledSFD(manifest(source.name), source_path=tmp_path / 'retrain.yaml')
    result = compiled.execute(experiment_dir=tmp_path / 'trained', progress_bar=False)
    trainer = Trainer(sfd=compiled, result=result)
    members = trainer.train(permutation_ids=result.data['_trial_id'].tolist(), validate_metrics=True)
    x, y = load_iris(return_X_y=True)
    prediction = members[0].predict(x)
    assert members[0].round_params['c'] == 0.5
    assert not any(trainer.validation.values())
    log = Log(uel_object=result, predictions=prediction, targets=y)
    table = log.prediction_table()
    assert len(table) == len(y)
    assert table['hit'].mean() > 0.9
    quality = log.data_quality(x)
    assert quality['rows'] == 150 and quality['missing'] == 0 and quality['infinite'] == 0
    diagnostics = log.confusion_value_diagnostics(x[:, 0])
    assert len(diagnostics) == 3 and 'tp_fp_ks' in diagnostics


def test_parquet_artifact_and_invalid_pruning_dry_run(project, monkeypatch):
    from talos.log import Log
    root, path = project
    monkeypatch.chdir(root)
    document = manifest()
    document['uel']['output_format'] = 'parquet'
    write_manifest(path, document)
    runner = CliRunner()
    result = runner.invoke(cli, ['run', '--no-progress-bar', str(path)])
    assert result.exit_code == 0, result.output
    artifact = next(root.glob('results/dev/*/results.parquet'))
    assert Log(file_path=artifact).experiment_log['score'].tolist() == [1, 2]
    document['uel']['pruning_strategies'] = [{'type': 'missing'}]
    write_manifest(path, document)
    result = runner.invoke(cli, ['run', '--dry-run', str(path)])
    assert result.exit_code == 1 and 'pruning' in result.output.lower()


def test_cohort_structured_multioutput_and_custom_aggregation_real_iris():
    from types import SimpleNamespace

    import pandas as pd
    from sklearn.datasets import load_iris

    from talos.cohort import Cohort
    x, _ = load_iris(return_X_y=True)
    result = SimpleNamespace(data=pd.DataFrame({'_trial_id': ['a', 'b']}),
                             details=pd.Series({'identity_hash': 'iris'}), run_dir=None,
                             metadata={}, round_history=[])
    cohort = Cohort(result=result, aggregation=lambda arrays: arrays.mean(axis=0))

    class Member:
        def __init__(self, identifier):
            self.permutation_id = identifier

        def predict(self, inputs):
            return {'features': inputs, 'measurements': (inputs[:, 0], inputs[:, 1])}
    cohort.set_members([Member('a'), Member('b')])
    output = cohort.predict(x)
    np.testing.assert_allclose(output['features'], x)
    np.testing.assert_allclose(output['measurements'][0], x[:, 0])


def test_source_bundle_preserves_local_packages_and_callable_candidates(tmp_path):
    import sys

    from talos.experiment.serialization import decode, dumps
    from talos.experiment.source_snapshot import hydrate_sources, snapshot_sources
    from talos.yaml.resolver import load_sfd
    package = tmp_path / 'owned_support'
    package.mkdir()
    (package / '__init__.py').write_text('from .values import candidate\n')
    helper = package / 'values.py'
    helper.write_text('def candidate(): return 7\n')
    source = tmp_path / 'caller_bundle.py'
    source.write_text('from owned_support import candidate\n'
                      'def params(): return {"fn": [candidate]}\n'
                      'def prep(context, round_params): return context\n'
                      'def model(context, round_params): return {"score": round_params["fn"]()}\n')
    caller = load_sfd(str(source))
    directory = tmp_path / 'saved'
    bundle = snapshot_sources(caller.model, caller.prep, caller.params(), directory)
    assert 'owned_support' in bundle['modules'] and 'owned_support.values' in bundle['modules']
    assert all('site-packages' not in item['original_path'] for item in bundle['modules'].values())
    encoded = dumps(caller.params())
    helper.write_text('def candidate(): return 700\n')
    for name in ['owned_support', 'owned_support.values']:
        sys.modules.pop(name, None)
    load_sfd(str(source))
    hydrate_sources(bundle, directory)
    assert decode(json.loads(encoded))['fn'][0]() == 7
    source.unlink()
    helper.unlink()
    (package / '__init__.py').unlink()
    (directory / 'source_bundle.json').write_text(json.dumps(bundle))
    (directory / 'params.json').write_text(encoded)
    script = ('import json; from talos.experiment.source_snapshot import hydrate_sources; '
              'from talos.experiment.serialization import decode; '
              f'root={str(directory)!r}; '
              'hydrate_sources(json.load(open(root+"/source_bundle.json")),root); '
              'assert decode(json.load(open(root+"/params.json")))["fn"][0]()==7')
    fresh = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert fresh.returncode == 0, fresh.stderr
    saved_helper = directory / bundle['modules']['owned_support.values']['path']
    saved_helper.write_text('def candidate(): return 99\n')
    with pytest.raises(ValueError, match='checksum'):
        hydrate_sources(bundle, directory)


def test_source_bundle_namespace_package_and_static_lazy_import(tmp_path, monkeypatch):
    from talos.experiment.source_snapshot import hydrate_sources, snapshot_sources
    from talos.yaml.resolver import load_sfd
    namespace = tmp_path / 'owned_namespace'
    namespace.mkdir()
    (namespace / 'helper.py').write_text('def value(): return 11\n')
    source = tmp_path / 'caller_namespace.py'
    source.write_text('def params(): return {"n": [1]}\n'
                      'def prep(context, round_params): return context\n'
                      'def model(context, round_params):\n'
                      '    from owned_namespace.helper import value\n'
                      '    return {"score": value()}\n')
    caller = load_sfd(str(source))
    bundle = snapshot_sources(caller.model, caller.prep, caller.params(), tmp_path / 'saved')
    assert 'owned_namespace.helper' in bundle['modules']
    assert 'owned_namespace' in bundle['modules']
    monkeypatch.syspath_prepend(str(tmp_path))
    assert caller.model(None, {'n': 1})['score'] == 11
    source.unlink()
    (namespace / 'helper.py').unlink()
    hydrate_sources(bundle, tmp_path / 'saved')
    import sys
    restored = sys.modules[caller.model.__module__]
    assert restored.model(None, {'n': 1})['score'] == 11


def test_saved_result_source_bundle_restore_without_original_main_or_helpers(tmp_path):
    import sys

    from talos.experiment.runner import run
    support = tmp_path / 'saved_support'
    support.mkdir()
    (support / '__init__.py').write_text('from .candidate import candidate\n')
    (support / 'candidate.py').write_text('def candidate(): return 19\n')
    source = tmp_path / 'source_saved.py'
    source.write_text('from saved_support import candidate\n'
                      'def params(): return {"fn": [candidate]}\n'
                      'def prep(context, round_params): return context\n'
                      'def model(context, round_params): return {"score": round_params["fn"]()}\n')
    result = run(str(source), experiment_dir=tmp_path / 'saved', progress_bar=False)
    raw = json.loads((result.run_dir / 'metadata.json').read_text())
    assert 'source_bundle' in raw
    assert 'saved_support.candidate' in raw['source_bundle']['modules']
    source.unlink()
    (support / '__init__.py').unlink()
    (support / 'candidate.py').unlink()
    script = ('from talos.experiment.runner import RunResult; '
              f'result=RunResult.load({str(result.run_dir)!r}); '
              'assert result.params["fn"][0]()==19; assert result.data.score.tolist()==[19]')
    restored = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert restored.returncode == 0, restored.stderr
    helper = result.run_dir / raw['source_bundle']['modules']['saved_support.candidate']['path']
    helper.write_text('def candidate(): return 190\n')
    corrupted = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert corrupted.returncode != 0 and 'checksum' in corrupted.stderr.lower()


def test_sensor_and_trainer_original_params_survive_metric_and_timestamp_aliases(tmp_path):
    from talos.experiment.runner import RunResult
    from talos.inference import Sensor, Trainer
    source = tmp_path / 'aliased_iris.py'
    source.write_text(
        "from sklearn.datasets import load_iris\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "def params(): return {'loss': ['crossentropy'], 'start': [99]}\n"
        "def prep(context, round_params): return load_iris(return_X_y=True)\n"
        "def model(prepared, round_params):\n"
        "    assert round_params == {'loss': 'crossentropy', 'start': 99}\n"
        "    x, y = prepared\n"
        "    classifier = LogisticRegression(max_iter=300).fit(x, y)\n"
        "    return {'loss': 0.25, 'model': classifier}\n")
    config = manifest(source.name)
    config['sfd']['objective'] = {'metric': 'loss', 'direction': 'min'}
    compiled = CompiledSFD(config, source_path=tmp_path / 'aliased.yaml')
    result = compiled.execute(experiment_dir=tmp_path / 'run', progress_bar=False)
    assert result.data['loss'].tolist() == [0.25]
    assert result.data[result.parameter_columns['loss']].tolist() == ['crossentropy']
    loaded = RunResult.load(result.run_dir)
    identifier = loaded.data['_trial_id'].iloc[0]
    for sensor in [Sensor(result=loaded, permutation_id=identifier), Sensor(result=loaded)]:
        assert sensor.round_params == {'loss': 'crossentropy', 'start': 99}
    trainer = Trainer(sfd=compiled, result=loaded)
    members = trainer.train(permutation_ids=[identifier], validate_metrics=True)
    assert members[0].round_params == {'loss': 'crossentropy', 'start': 99}


def test_source_hydration_preserves_identical_callables_and_unbundled_package_modules(tmp_path, monkeypatch):
    import importlib
    import sys

    from talos.experiment.source_snapshot import hydrate_sources, snapshot_sources
    package = tmp_path / 'caller_cache'
    package.mkdir()
    (package / '__init__.py').write_text('')
    (package / 'sfd.py').write_text('def model(data, params): return {"score": 1}\n')
    (package / 'unrelated.py').write_text('def unrelated(): return 7\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    source = importlib.import_module('caller_cache.sfd')
    unrelated = importlib.import_module('caller_cache.unrelated')
    model = source.model
    directory = tmp_path / 'run'
    bundle = snapshot_sources(model, None, {}, directory)
    assert 'caller_cache.unrelated' not in bundle['modules']
    hydrate_sources(bundle, directory)
    assert sys.modules['caller_cache.sfd'] is source
    assert sys.modules['caller_cache.sfd'].model is model
    assert sys.modules['caller_cache.unrelated'] is unrelated


def test_saved_result_rejects_missing_checkpoint_trial_records(project):
    from talos.experiment.runner import RunResult
    root, path = project
    result = CompiledSFD(manifest(), source_path=path).execute(experiment_dir=root / 'run', progress_bar=False)
    records = result.run_dir / 'round_data.jsonl'
    records.write_text(records.read_text().splitlines()[0] + '\n')
    with pytest.raises(ValueError, match='missing completed trial'):
        RunResult.load(result.run_dir)


def test_manifest_local_callable_and_lazy_helper_imports_outside_cwd(tmp_path, monkeypatch):
    from talos.yaml.profiler import profile
    project = tmp_path / 'caller_project'
    project.mkdir()
    source = project / 'scoped.py'
    source.write_text('def params(): return {"fn": [abs]}\n'
                      'def prep(context, round_params):\n'
                      '    from scoped_helper import data\n'
                      '    return data()\n'
                      'def model(context, round_params): return {"score": round_params["fn"]()}\n')
    (project / 'scoped_helper.py').write_text('import numpy as np\n'
                                            'def candidate(): return 23\n'
                                            'def data(): return np.array([1., float("nan"), float("inf")])\n')
    monkeypatch.chdir(tmp_path)
    config = manifest(source.name)
    config['sfd']['params'] = {'fn': [{'callable': 'scoped_helper:candidate'}]}
    compiled = CompiledSFD(config, source_path=project / 'manifest.yaml')
    diagnostics = profile(compiled)
    assert diagnostics.sample_permutations_completed == 1
    assert any('NaN' in warning for warning in diagnostics.data_quality_warnings)
    assert any('Inf' in warning for warning in diagnostics.data_quality_warnings)
    result = compiled.execute(experiment_dir=tmp_path / 'run', progress_bar=False)
    assert result.data['score'].tolist() == [23]


def test_cohort_callable_candidates_need_no_arrow_and_parameter_correlations_use_aliases(tmp_path):
    from talos.cohort import Cohort
    from talos.log import Log
    source = tmp_path / 'diagnostic_alias.py'
    source.write_text('def params(): return {"score": list(range(12)), "optimizer": [int]}\n'
                      'def prep(context, round_params): return context\n'
                      'def model(context, round_params):\n'
                      '    return {"score": round_params["score"] * 2, "auxiliary": round_params["score"] + 1}\n')
    config = manifest(source.name)
    config['uel']['round_limit'] = 12
    result = CompiledSFD(config, source_path=tmp_path / 'diagnostic.yaml').execute(
        experiment_dir=tmp_path / 'run', progress_bar=False)
    cohort = Cohort(result=result, selector='top_n', selector_params={'column': 'score', 'n': 2})
    assert len(cohort.permutation_ids) == 2
    correlations = Log(uel_object=result).experiment_parameter_correlation('score', heads=[1.], min_n=10, n_boot=5)
    assert set(correlations.index.get_level_values('feature')) == {result.parameter_columns['score']}


def test_fresh_cli_resume_deleted_sources_preserves_completed_models_and_parquet(tmp_path):
    import hashlib
    import os
    import sys

    import polars as pl
    project = tmp_path / 'portable_project'
    project.mkdir()
    source = project / 'portable_sfd.py'
    helper = project / 'portable_support.py'
    marker, calls = project / 'pause-once', project / 'trained.txt'
    marker.touch()
    helper.write_text('from sklearn.datasets import load_iris\n'
                      'def candidate(): return 0\n'
                      'def observations(): return load_iris(return_X_y=True)\n')
    source.write_text(
        'import os, signal\n'
        'from pathlib import Path\n'
        'from sklearn.linear_model import LogisticRegression\n'
        'from portable_support import candidate, observations\n'
        'def params(): return {"c": [0.5, 1.0], "optimizer": [candidate]}\n'
        'def prep(context, round_params): return context, observations()\n'
        'def model(prepared, round_params):\n'
        '    context, (x, y) = prepared\n'
        '    marker = Path(context["marker"])\n'
        '    if round_params["c"] == 1.0 and marker.exists():\n'
        '        marker.unlink()\n'
        '        os.kill(os.getpid(), signal.SIGTERM)\n'
        '    assert round_params["optimizer"]() == 0\n'
        '    classifier = LogisticRegression(C=round_params["c"], max_iter=300).fit(x, y)\n'
        '    with Path(context["calls"]).open("a") as output: output.write(str(round_params["c"]) + "\\n")\n'
        '    return {"score": classifier.score(x, y), "model": classifier}\n')
    config = manifest(source.name)
    config['sfd']['context'] = {'marker': str(marker), 'calls': str(calls)}
    config['uel']['output_format'] = 'parquet'
    path = project / 'manifest.yaml'
    write_manifest(path, config)
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[1]),
                                                    env.get('PYTHONPATH', '')]))
    paused = subprocess.run([sys.executable, '-m', 'talos', 'run', '--no-progress-bar', str(path)],
                            cwd=project, env=env, capture_output=True, text=True)
    assert paused.returncode == 0, paused.stdout + paused.stderr
    directory = next(project.glob('results/dev/*'))
    raw = json.loads((directory / 'metadata.json').read_text())
    assert raw['details']['status'] == 'paused' and raw['details']['rounds'] == 1
    first = json.loads((directory / 'round_data.jsonl').read_text().splitlines()[0])
    artifact = directory / first['artifact']['path']
    original_weight = hashlib.sha256(artifact.read_bytes()).hexdigest(), artifact.stat().st_mtime_ns
    assert pl.read_parquet(directory / 'results.parquet').height == 1
    source.unlink()
    helper.unlink()
    path.unlink()
    resumed = subprocess.run([sys.executable, '-m', 'talos', 'run', '--resume', str(directory), '--no-progress-bar'],
                             cwd=tmp_path, env=env, capture_output=True, text=True)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    current = json.loads((directory / 'metadata.json').read_text())
    records = [json.loads(line) for line in (directory / 'round_data.jsonl').read_text().splitlines()]
    assert current['details']['status'] == 'complete' and current['details']['rounds'] == 2
    assert [record['params']['c'] for record in records] == [0.5, 1.0]
    assert records[0] == first
    assert records[0]['trial_id'] != records[1]['trial_id']
    assert calls.read_text().splitlines() == ['0.5', '1.0']
    assert (hashlib.sha256(artifact.read_bytes()).hexdigest(), artifact.stat().st_mtime_ns) == original_weight
    assert current['identity_hash'] == raw['identity_hash']
    assert current['source_bundle'] == raw['source_bundle']
    assert pl.read_parquet(directory / 'results.parquet')['c'].to_list() == [0.5, 1.0]
    saved_helper = directory / raw['source_bundle']['modules']['portable_support']['path']
    saved_helper.write_text(saved_helper.read_text() + '\n# checksum mismatch\n')
    rejected = subprocess.run([sys.executable, '-m', 'talos', 'run', '--resume', str(directory), '--no-progress-bar'],
                              cwd=tmp_path, env=env, capture_output=True, text=True)
    assert rejected.returncode == 1 and 'checksum' in rejected.stdout.lower()
    assert (hashlib.sha256(artifact.read_bytes()).hexdigest(), artifact.stat().st_mtime_ns) == original_weight


def test_cli_resume_rejects_changed_surviving_helper_source(tmp_path):
    source = tmp_path / 'surviving_sfd.py'
    helper = tmp_path / 'surviving_support.py'
    helper.write_text('def candidate(): return 1\n')
    source.write_text('from surviving_support import candidate\n'
                      'def params(): return {"fn": [candidate]}\n'
                      'def prep(context, round_params): return context\n'
                      'def model(context, round_params): return {"score": round_params["fn"]()}\n')
    path = tmp_path / 'manifest.yaml'
    result = CompiledSFD(manifest(source.name), source_path=path).execute(
        experiment_dir=tmp_path / 'run', progress_bar=False)
    helper.write_text('def candidate(): return 2\n')
    resumed = CliRunner().invoke(cli, ['run', '--resume', str(result.run_dir), '--no-progress-bar'])
    assert resumed.exit_code == 1 and 'source changed' in resumed.output.lower()
    assert len((result.run_dir / 'round_data.jsonl').read_text().splitlines()) == 1


@pytest.mark.parametrize('kind', ['factory', 'object'])
def test_manifest_factory_and_object_compile_once_and_portable_cli_resume(tmp_path, kind):
    import os
    import sys
    helper = tmp_path / f'manifest_support_{kind}.py'
    helper.write_text('import polars as pl\nfrom sklearn.datasets import load_iris\n'
                      'def observations():\n'
                      '    x, y = load_iris(return_X_y=True)\n'
                      '    return pl.DataFrame(x, schema=["a", "b", "c", "d"]).with_columns(pl.Series("species", y))\n')
    source = tmp_path / 'manifest_only.py'
    construction = ('factories = 0\ndef manifest():\n'
                    '    global factories\n    factories += 1\n'
                    '    return IrisManifest().set_target_column("species").with_reference_architecture(architecture)\n'
                    if kind == 'factory' else
                    'manifest = IrisManifest().set_target_column("species").with_reference_architecture(architecture)\n')
    source.write_text(
        'from talos.experiment.manifest_core import MLManifest\n'
        'from sklearn.linear_model import LogisticRegression\n'
        f'from manifest_support_{kind} import observations\n'
        'def params(): return {"c": [0.1, 1.0]}\n'
        'class IrisManifest(MLManifest):\n'
        '    def prepare_data(self, data, round_params):\n'
        '        return super().prepare_data(observations() if data is None else data, round_params)\n'
        'def architecture(data, c):\n'
        '    model = LogisticRegression(C=c, max_iter=300).fit(data["x_train"], data["y_train"])\n'
        '    return {"score": model.score(data["x_val"], data["y_val"]), "model": model}\n'
        + construction)
    compiled = CompiledSFD(manifest(source.name), source_path=tmp_path / 'experiment.yaml')
    if kind == 'factory':
        assert compiled._source.factories == 1
    result = compiled.execute(experiment_dir=tmp_path / 'run', progress_bar=False, stop_after=1)
    assert result.status == 'paused' and len(result.data) == 1
    if kind == 'factory':
        assert compiled._source.factories == 1
    assert result.metadata['identity']['preparation_manifest']['fields']['target_column'] == 'species'
    assert result.metadata['sfd']['module'] == compiled._source.__name__
    source.unlink()
    helper.unlink()
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(Path(__file__).resolve().parents[1]),
                                                    env.get('PYTHONPATH', '')]))
    resumed = subprocess.run([sys.executable, '-m', 'talos', 'run', '--resume', str(result.run_dir), '--no-progress-bar'],
                             env=env, capture_output=True, text=True)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    raw = json.loads((result.run_dir / 'metadata.json').read_text())
    records = [json.loads(line) for line in (result.run_dir / 'round_data.jsonl').read_text().splitlines()]
    assert raw['details']['status'] == 'complete' and raw['details']['rounds'] == 2
    assert [record['params']['c'] for record in records] == [0.1, 1.0]
    assert raw['identity_hash'] == result.metadata['identity_hash']
    from talos.inference import Trainer
    retrained = Trainer(run_dir=result.run_dir).train(permutation_ids=[records[0]['trial_id']], validate_metrics=True)
    assert retrained[0].round_params['c'] == 0.1
    assert len((result.run_dir / 'round_data.jsonl').read_text().splitlines()) == 2


def test_same_iris_sfd_tuple_ranges_native_manifest_and_resume(tmp_path):
    from talos.experiment.runner import run
    source = tmp_path / 'iris_tuple_ranges.py'
    source.write_text(
        "from sklearn.datasets import load_iris\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.model_selection import train_test_split\n"
        "def params(): return {'units': (4, 10, 3)}\n"
        "def prep(data, round_params):\n"
        "    x, y = load_iris(return_X_y=True)\n"
        "    return train_test_split(x, y, stratify=y, random_state=17)\n"
        "def model(prepared, round_params):\n"
        "    x, xv, y, yv = prepared\n"
        "    model = LogisticRegression(C=round_params['units'], max_iter=300).fit(x, y)\n"
        "    return {'score': model.score(xv, yv), 'model': model}\n")
    document = manifest(source.name)
    document['uel']['round_limit'] = 3
    compiled = CompiledSFD(document, source_path=tmp_path / 'experiment.yaml')
    assert compiled.params() == {'units': [4, 6, 8]}
    native = run(compiled._source, experiment_dir=tmp_path / 'native', seed=42,
                 round_limit=3, progress_bar=False)
    paused = compiled.execute(experiment_dir=tmp_path / 'compiled', progress_bar=False, stop_after=1)
    assert paused.status == 'paused' and paused.data.units.tolist() == [4]
    recorded = (paused.run_dir / 'round_data.jsonl').read_text().splitlines()[0]
    result = CompiledSFD.from_run(paused.run_dir).execute(
        resume=True, experiment_dir=paused.run_dir, progress_bar=False)
    assert native.data.units.tolist() == result.data.units.tolist() == [4, 6, 8]
    assert native.data['_trial_id'].tolist() == result.data['_trial_id'].tolist()
    np.testing.assert_array_equal(native.data.score, result.data.score)
    assert (paused.run_dir / 'round_data.jsonl').read_text().splitlines()[0] == recorded
    overridden = manifest(source.name)
    overridden['sfd']['params'] = {'units': [4, 10, 3]}
    assert CompiledSFD(overridden, source_path=tmp_path / 'override.yaml').params() == {'units': [4, 10, 3]}


@pytest.mark.parametrize('metadata_kind', ['wheel', 'egg-sources', 'egg-installed'])
def test_source_bundle_excludes_target_installed_distribution(tmp_path, monkeypatch, metadata_kind):
    import importlib
    import sys

    from talos.experiment.source_snapshot import hydrate_sources, snapshot_sources
    from talos.yaml.resolver import load_sfd
    for name in ('alternate_dependency', 'alternate_dependency.integration'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    target = tmp_path / 'alternate_dependencies'
    package = target / 'alternate_dependency'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text(
        'def value(): return 17\n'
        'def optional():\n'
        '    from .integration import missing\n'
        '    return missing()\n')
    (package / 'integration.py').write_text('from absent_optional_backend import missing\n')
    metadata = target / ('alternate_dependency-1.0.dist-info' if metadata_kind == 'wheel'
                         else 'alternate_dependency.egg-info')
    metadata.mkdir()
    (metadata / ('METADATA' if metadata_kind == 'wheel' else 'PKG-INFO')).write_text(
        'Metadata-Version: 2.1\nName: alternate-dependency\nVersion: 1.0\n')
    if metadata_kind == 'wheel':
        (metadata / 'RECORD').write_text(
            'alternate_dependency/__init__.py,,\n'
            'alternate_dependency/integration.py,,\n'
            f'{metadata.name}/METADATA,,\n'
            f'{metadata.name}/RECORD,,\n')
    elif metadata_kind == 'egg-sources':
        (metadata / 'SOURCES.txt').write_text(
            'alternate_dependency/__init__.py\nalternate_dependency/integration.py\n')
    else:
        (metadata / 'installed-files.txt').write_text(
            '../alternate_dependency/__init__.py\n../alternate_dependency/integration.py\n')
    monkeypatch.syspath_prepend(str(target))
    importlib.invalidate_caches()
    source = tmp_path / 'caller_target_dependency.py'
    source.write_text('from alternate_dependency import value\n'
                      'def params(): return {"value": [value]}\n'
                      'def prep(data, round_params): return data\n'
                      'def model(data, round_params): return {"score": round_params["value"]()}\n')
    caller = load_sfd(str(source))
    directory = tmp_path / 'saved'
    bundle = snapshot_sources(caller.model, caller.prep, caller.params(), directory)
    assert set(bundle['modules']) == {caller.__name__}
    assert all(not Path(item['original_path']).is_relative_to(target)
               for item in bundle['modules'].values())
    source.unlink()
    hydrate_sources(bundle, directory)
    assert importlib.import_module(caller.__name__).model(None, {'value': caller.params()['value'][0]}) == {'score': 17}


def test_source_bundle_resolves_full_module_namespace_without_leaf_alias(tmp_path, monkeypatch):
    import importlib

    from talos.experiment.source_snapshot import hydrate_sources, snapshot_sources
    package = tmp_path / 'qualified_caller'
    package.mkdir()
    (package / '__init__.py').write_text('')
    (package / 'same_leaf.py').write_text('def value(): return 13\n')
    (package / 'sfd.py').write_text(
        'def params(): return {"n": [1]}\n'
        'def model(data, round_params):\n'
        '    from .same_leaf import value\n'
        '    return {"score": value()}\n'
        'def unused_optional():\n'
        '    import same_leaf\n'
        '    return same_leaf.value()\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    caller = importlib.import_module('qualified_caller.sfd')
    directory = tmp_path / 'saved'
    bundle = snapshot_sources(caller.model, None, caller.params(), directory)
    assert set(bundle['modules']) == {'qualified_caller', 'qualified_caller.sfd', 'qualified_caller.same_leaf'}
    assert 'same_leaf' not in bundle['modules']
    for path in package.glob('*.py'):
        path.unlink()
    hydrate_sources(bundle, directory)
    assert importlib.import_module('qualified_caller.sfd').model(None, {'n': 1}) == {'score': 13}
