import numpy as np
import polars as pl
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from talos.experiment.manifest_core import MachineLearningManifest, Manifest, MLManifest
from talos.scalers.robust_scaler import RobustScaler


def iris_splits():
    iris = load_iris()
    frame = pl.DataFrame({name: iris.data[:, index] for index, name in enumerate(
        ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])}).with_columns(pl.Series('species', iris.target))
    train, heldout = train_test_split(np.arange(len(iris.target)), test_size=.4, random_state=17, stratify=iris.target)
    val, test = train_test_split(heldout, test_size=.5, random_state=19, stratify=iris.target[heldout])
    return [frame[indices] for indices in (train, val, test)]


def test_scaler_is_fit_only_on_real_training_observations():
    splits = iris_splits()
    manifest = MLManifest().set_target_column('species').set_scaler(RobustScaler)
    output = manifest.prepare_data(splits, {})
    scaler = output['_scaler']
    assert set(scaler.medians) == {'sepal_length', 'sepal_width', 'petal_length', 'petal_width'}
    assert scaler.medians['sepal_length'] == splits[0]['sepal_length'].median()
    expected = scaler.transform(splits[1].drop('species')).to_numpy()
    assert np.array_equal(output['x_val'], expected)
    assert np.array_equal(output['y_val'], splits[1]['species'].to_numpy())
    swapped = manifest.prepare_data([splits[0], splits[2], splits[1]], {})
    assert scaler.medians == swapped['_scaler'].medians
    assert scaler.iqrs == swapped['_scaler'].iqrs
    assert np.array_equal(output['x_train'], swapped['x_train'])
    assert not hasattr(manifest, 'data_source_config')
    assert not hasattr(manifest, 'fetch_data')


def test_pca_state_is_train_only_and_reused_on_heldout():
    splits = iris_splits()
    manifest = MLManifest().set_target_column('species').set_scaler(RobustScaler).set_pca_compression()
    output = manifest.prepare_data(splits, {'auto_pca': True, 'pca_k': 2})
    scaled_train = output['_scaler'].transform(splits[0].drop('species')).to_numpy()
    assert np.allclose(output['_pca'].mean_, scaled_train.mean(axis=0))
    scaled_test = output['_scaler'].transform(splits[2].drop('species')).to_numpy()
    assert np.allclose(output['x_test'], output['_pca'].transform(scaled_test))
    swapped = manifest.prepare_data([splits[0], splits[2], splits[1]], {'auto_pca': True, 'pca_k': 2})
    assert np.array_equal(output['_pca'].components_, swapped['_pca'].components_)
    assert output['_feature_names'] == ['pc_0', 'pc_1']


def test_fitted_transform_computation_receives_training_once():
    splits = iris_splits()
    calls = []

    def fit_mean(data):
        calls.append(data.height)
        return data['sepal_length'].mean()

    def center(data, mean):
        return data.with_columns((pl.col('sepal_length') - mean).alias('sepal_length'))
    manifest = MLManifest().set_target_column('species').add_fitted_transform(
        [('train_mean', fit_mean, {})], center, mean='train_mean')
    output = manifest.prepare_data(splits, {})
    assert calls == [len(splits[0])]
    assert output['train_mean'] == splits[0]['sepal_length'].mean()
    assert np.allclose(output['x_test'][:, 0], splits[2]['sepal_length'].to_numpy() - output['train_mean'])


def test_target_transform_is_fit_on_training_only():
    splits = iris_splits()
    calls = []

    class CenteredTarget:
        def __init__(self, train_data, target_name):
            calls.append(train_data.height)
            self.mean = train_data['species'].mean()
            self.target_name = target_name

        def transform(self, data):
            return data.with_columns((pl.col('species') - self.mean).alias(self.target_name)).drop('species')
    manifest = MLManifest().with_target_label('centered_species', CenteredTarget)
    output = manifest.prepare_data(splits, {})
    assert calls == [len(splits[0])]
    target = output['_target_cls_centered_species']
    assert target.mean == splits[0]['species'].mean()
    assert np.allclose(output['y_test'], splits[2]['species'].to_numpy() - target.mean)


def test_resolution_splitter_and_deep_copy_preserve_source_manifest():
    splits = iris_splits()
    frame = pl.concat(splits)
    manifest = MLManifest().set_target_column('species').set_split_config(3, 1, 1)
    overridden = manifest.with_params_override(split_config=(1, 0, 0), scale=2)
    assert manifest.split_config == (3, 1, 1)
    assert overridden.split_config == (1, 0, 0)
    assert manifest.architecture_params == {}
    assert overridden.architecture_params == {'scale': 2}
    output = overridden.prepare_data(frame, {})
    assert len(output['x_train']) == len(frame)
    assert output['x_val'].shape == (0, 4)
    assert output['x_test'].shape == (0, 4)
    assert MachineLearningManifest is MLManifest
    with pytest.raises(TypeError, match='Supply'):
        manifest.prepare_data(None, {})


def test_generic_transform_group_and_parameter_resolution():
    splits = iris_splits()

    def scale_column(data, column, factor):
        return data.with_columns((pl.col(column) * factor).alias(column))
    manifest = Manifest().set_target_column('species').add_transform(
        scale_column, group='scale', include_if='enabled', column='sepal_length', factor='{factor}')
    output = manifest.prepare_data(splits, {'enabled': True, 'factor': 2, 'feature_groups': 'scale'})
    assert np.array_equal(output['x_train'][:, 0], splits[0]['sepal_length'].to_numpy() * 2)
    disabled = manifest.prepare_data(splits, {'enabled': False, 'factor': 2})
    assert np.array_equal(disabled['x_train'][:, 0], splits[0]['sepal_length'].to_numpy())


def test_calibration_builder_resolves_without_leaking_test_into_fit():
    splits = iris_splits()
    calls = []

    def calibrator(model, x_val, y_val, method):
        calls.append((len(x_val), method))
        return model

    def model(data, prediction_calibration_config):
        config = prediction_calibration_config
        config.calibration_func(None, data['x_val'], data['y_val'], **config.calibration_params)
        return {'validation_rows': len(data['x_val'])}
    manifest = MLManifest().set_target_column('species').with_reference_architecture(model)
    manifest.with_calibration().probability_calibration(calibrator, method='calibration_method').done()
    data = manifest.prepare_data(splits, {})
    output = manifest.run_model(data, {'calibration_method': 'sigmoid'})
    assert calls == [(len(splits[1]), 'sigmoid')]
    assert output['validation_rows'] == len(splits[1])
    assert manifest.prediction_calibration_config.calibration_params == {'method': 'calibration_method'}


def test_scaler_selection_and_ablation_are_consistent_across_splits():
    splits = iris_splits()
    manifest = MLManifest().set_target_column('species').set_scaler_from_params().set_feature_ablation()
    output = manifest.prepare_data(splits, {'scaler_type': 'robust', 'feature_drop_count': 1, 'feature_drop_seed': 17})
    assert output['x_train'].shape[1] == output['x_val'].shape[1] == output['x_test'].shape[1] == 3
    assert len(output['_dropped_features']) == 1
    assert 'species' not in output['_dropped_features']
    assert len(output['_scaler'].medians) == 4


def test_caller_owned_random_split_defaults_and_seed_references():
    frame = pl.concat(iris_splits())
    default = Manifest().set_target_column('species').set_random_split()
    assert sum(len(default.prepare_data(frame, {})['x_' + name]) for name in ['train', 'val', 'test']) == len(frame)
    seeded = Manifest().set_target_column('species').set_random_split(seed='round_seed')
    first = seeded.prepare_data(frame, {'round_seed': 17})
    second = seeded.prepare_data(frame, {'round_seed': 17})
    assert np.array_equal(first['x_train'], second['x_train'])


def test_calibration_flags_supply_disabled_config_to_required_argument():
    def model(data, prediction_calibration_config):
        return prediction_calibration_config
    manifest = MLManifest().with_reference_architecture(model)
    manifest.with_calibration().probability_calibration(lambda *args: None).done()
    config = manifest.run_model({}, {'use_calibration': False, 'use_threshold': False})
    assert config.calibration_func is None
    assert config.threshold_func is None
    with pytest.raises(TypeError, match='boolean'):
        manifest.run_model({}, {'use_calibration': 'false'})


def test_multioutput_targets_and_strict_finite_values():
    splits = iris_splits()
    manifest = MLManifest().set_target_columns(['species', 'petal_width'])
    output = manifest.prepare_data(splits, {})
    assert output['x_train'].shape == (len(splits[0]), 3)
    assert output['y_train'].shape == (len(splits[0]), 2)
    assert output['_target_names'] == ['species', 'petal_width']
    strict = MLManifest().set_target_column('species').set_strict_mode(True)
    invalid = splits[1].with_columns(pl.when(pl.arange(0, pl.len()) == 0).then(float('nan')).otherwise(pl.col('sepal_length')).alias('sepal_length'))
    with pytest.raises(ValueError, match='non-finite'):
        strict.prepare_data([splits[0], invalid, splits[2]], {})


def test_configuration_records_pipeline_without_reconstructing_fitted_closures():
    import json

    from talos.experiment.serialization import content_hash, decode, dumps
    manifest = MLManifest().set_target_column('species').set_scaler(RobustScaler)
    config = manifest.configuration()
    assert config['fields']['target_column'] == 'species'
    assert config['fields']['scaler']['__talos_type__'] == 'tuple'
    assert decode(json.loads(dumps(config))) == config
    before = content_hash(config)
    manifest.prepare_data(iris_splits(), {})
    assert content_hash(manifest.configuration()) == before
    changed = manifest.with_params_override(split_config=(1, 0, 0))
    assert content_hash(changed.configuration()) != before


def test_datetime_split_configuration_remains_canonical_data():
    import json
    from datetime import datetime

    from talos.experiment.serialization import decode, dumps
    manifest = MLManifest().set_split_dates(datetime(2020, 1, 1), datetime(2021, 1, 1),
        datetime(2021, 1, 1), datetime(2022, 1, 1), datetime(2022, 1, 1), datetime(2023, 1, 1))
    config = manifest.configuration()
    assert decode(json.loads(dumps(config))) == config


def _iris_manifest_architecture(data, c):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    model = LogisticRegression(C=c, max_iter=300).fit(data['x_train'], data['y_train'])
    probabilities = model.predict_proba(data['x_val'])
    return {'val_loss': log_loss(data['y_val'], probabilities), 'predictions': probabilities}


class ManifestFactoryIrisSFD:
    @staticmethod
    def params():
        return {'c': [.1, 1.]}

    @staticmethod
    def manifest():
        return (MLManifest().set_target_column('species').set_scaler(RobustScaler)
                .with_reference_architecture(_iris_manifest_architecture))


def test_manifest_factory_only_sfd_runs_through_native_and_universal_executor(tmp_path):
    from talos.experiment import UniversalExperimentLoop, run
    splits = iris_splits()
    native = run(ManifestFactoryIrisSFD, splits, seed=17, progress_bar=False,
                 experiment_dir=tmp_path / 'native', save_models=False, save_weights=False,
                 objective={'metric': 'val_loss', 'direction': 'min'})
    facade = UniversalExperimentLoop(sfd=ManifestFactoryIrisSFD, data=splits, seed=17,
                 experiment_dir=tmp_path / 'facade', save_models=False, save_weights=False,
                 objective={'metric': 'val_loss', 'direction': 'min'})
    universal = facade.run('iris', n_permutations=2, progress_bar=False)
    expected = [_iris_manifest_architecture(ManifestFactoryIrisSFD.manifest().prepare_data(splits, {}), value)['val_loss']
                for value in [.1, 1.]]
    assert native.data.c.tolist() == universal.data.c.tolist() == [.1, 1.]
    assert np.allclose(native.data.val_loss, expected)
    assert np.array_equal(native.data.val_loss, universal.data.val_loss)
    assert native.data._trial_id.tolist() == universal.data._trial_id.tolist()
    assert native.metadata['identity']['preparation_manifest']['fields']['target_column'] == 'species'


def test_source_functions_include_user_scaler_hidden_in_factory_defaults():
    class UserScaler(RobustScaler):
        def transform(self, data):
            return super().transform(data)
    manifest = MLManifest().set_scaler(UserScaler).with_reference_architecture(_iris_manifest_architecture)
    functions = manifest.source_functions()
    assert UserScaler in functions
    assert _iris_manifest_architecture in functions
    assert len({id(function) for function in functions}) == len(functions)
