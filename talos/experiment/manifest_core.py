"""Caller-owned preparation and training manifests, forked from Limen."""
from __future__ import annotations
import copy
import inspect
import logging
import random
import re
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Callable

import numpy as np
import polars as pl
from sklearn.decomposition import PCA

from talos.calibration.pipeline import CalibratorProtocol, ThresholdOptimizerProtocol
from talos.preparation import split_sequential, split_random, split_by_dates, split_data_to_prep_output
from talos.scalers.registry import SCALER_REGISTRY
from talos.experiment.serialization import encode

logger = logging.getLogger(__name__)
ParamValue = Any
FittedTransformEntry = tuple

@dataclass
class TransformEntry:
    func: Callable[..., Any]
    params: dict[str, ParamValue] = field(default_factory=dict[str, ParamValue])
    group: str | None = None
    include_if: str | None = None

@dataclass
class AblationConfig:
    drop_count_key: str
    seed_key: str

@dataclass
class PCACompressionConfig:
    enabled_param: str
    n_components_param: str
    scaler_param_name: str
    component_prefix: str

@dataclass
class TargetClassConfig:
    target_class: type
    fit_params: dict[str, ParamValue] = field(default_factory=dict[str, ParamValue])
    transform_params: dict[str, ParamValue] = field(default_factory=dict[str, ParamValue])

@dataclass
class CalibrationConfig:
    calibration_func: CalibratorProtocol | None = None
    calibration_params: dict[str, Any] = field(default_factory=dict[str, Any])
    threshold_func: ThresholdOptimizerProtocol | None = None
    threshold_params: dict[str, Any] = field(default_factory=dict[str, Any])

    def resolve(self, round_params: dict[str, Any]) -> 'CalibrationConfig':
        return CalibrationConfig(calibration_func=self.calibration_func, calibration_params=_resolve_params(self.calibration_params, round_params), threshold_func=self.threshold_func, threshold_params=_resolve_params(self.threshold_params, round_params))

class CalibrationBuilder:

    def __init__(self, manifest: object) -> None:
        super().__init__()
        if not isinstance(manifest, MLManifest):
            raise ValueError(f'CalibrationBuilder requires an MLManifest, got {type(manifest).__name__}. Use MLManifest().with_calibration() to configure calibration.')
        self._manifest = manifest
        self._calibration_func: CalibratorProtocol | None = None
        self._calibration_params: dict[str, Any] = {}
        self._threshold_func: ThresholdOptimizerProtocol | None = None
        self._threshold_params: dict[str, Any] = {}

    def probability_calibration(self, func: CalibratorProtocol, **params: Any) -> 'CalibrationBuilder':
        self._calibration_func = func
        self._calibration_params = params
        return self

    def threshold_function(self, func: ThresholdOptimizerProtocol, **params: Any) -> 'CalibrationBuilder':
        self._threshold_func = func
        self._threshold_params = params
        return self

    def done(self) -> 'MLManifest':
        if self._calibration_func is None and self._threshold_func is None:
            raise ValueError('CalibrationBuilder at least one of probability_calibration() or threshold_function() must be called before done()')
        self._manifest.prediction_calibration_config = CalibrationConfig(calibration_func=self._calibration_func, calibration_params=dict(self._calibration_params), threshold_func=self._threshold_func, threshold_params=dict(self._threshold_params))
        return self._manifest

def _apply_fitted_transform(data: pl.DataFrame, fitted_transform: Any) -> pl.DataFrame:
    return fitted_transform.transform(data)

def _split_extra_params(extra_params: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    _extra = dict(extra_params or {})
    _static = {k: v for k, v in _extra.items() if not isinstance(v, str)}
    _dynamic = {k: v for k, v in _extra.items() if isinstance(v, str)}
    return (_static, _dynamic)

def make_fitted_scaler(param_name: str, transform_class: Any, extra_params: dict[str, Any] | None=None) -> FittedTransformEntry:
    _static, _dynamic = _split_extra_params(extra_params)

    def _factory(data: 'pl.DataFrame', _cls: Any=transform_class, _p: dict[str, Any]=_static, **dyn: Any) -> Any:
        return _cls(data, **_p, **dyn)
    return ([(param_name, _factory, _dynamic)], _apply_fitted_transform, {'fitted_transform': param_name})

def _resolve_params(params: dict[str, Any], round_params: dict[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, str):
            if value in round_params:
                resolved[key] = round_params[value]
            elif value.startswith('_'):
                resolved[key] = value
            elif '{' in value and '}' in value:
                m = re.fullmatch('\\{(\\w+)\\}', value.strip())
                if m:
                    resolved[key] = round_params[m.group(1)]
                else:
                    resolved[key] = value.format(**round_params)
            else:
                resolved[key] = value
        else:
            resolved[key] = value
    return resolved

def _should_include_transform(entry: TransformEntry, round_params: dict[str, Any]) -> bool:
    if entry.include_if is not None:
        if entry.include_if not in round_params:
            return False
        flag = round_params[entry.include_if]
        if not isinstance(flag, bool):
            raise TypeError(f"round_params['{entry.include_if}'] must be a bool, got {flag!r}")
        if not flag:
            return False
    if entry.group is None:
        return True
    return _is_group_active(entry.group, round_params)

def _is_group_active(group: str, round_params: dict[str, Any]) -> bool:
    feature_groups = round_params.get('feature_groups')
    if feature_groups is None or feature_groups == 'all':
        return True
    if not isinstance(feature_groups, str):
        raise TypeError(f"round_params['feature_groups'] must be a string, got {type(feature_groups).__name__}")
    return group in feature_groups.split('|')

def _apply_fitted_transforms(transform_entries: list[FittedTransformEntry], data: pl.DataFrame, round_params: dict[str, Any], all_fitted_params: dict[str, Any], is_training: bool) -> tuple[pl.DataFrame, dict[str, Any]]:
    for fitted_param_computations, func, base_params in transform_entries:
        for param_name, compute_func, compute_base_params in fitted_param_computations:
            if param_name not in all_fitted_params and is_training:
                resolved = _resolve_params(compute_base_params, round_params)
                value = compute_func(data, **resolved)
                all_fitted_params[param_name] = value
        combined_round_params = {**round_params, **all_fitted_params}
        resolved = _resolve_params(base_params, combined_round_params)
        data = func(data, **resolved)
    return (data, all_fitted_params)

def _apply_class_based_target(manifest: Manifest, data: pl.DataFrame, round_params: dict[str, Any], all_fitted_params: dict[str, Any], is_training: bool) -> tuple[pl.DataFrame, dict[str, Any]]:
    config = manifest.target_class_config
    if config is None:
        raise ValueError('_apply_class_based_target manifest has no target_class_config')
    target_name = manifest.target_column
    instance_key = f'_target_cls_{target_name}'
    if is_training:
        resolved_fit = _resolve_params(config.fit_params, round_params)
        instance = config.target_class(train_data=data, target_name=target_name, **resolved_fit)
        all_fitted_params[instance_key] = instance
    else:
        if instance_key not in all_fitted_params:
            raise RuntimeError(f"Target instance '{instance_key}' not found — training split must run before validation/test.")
        instance = all_fitted_params[instance_key]
    resolved_transform = _resolve_params(config.transform_params, round_params)
    data = instance.transform(data, **resolved_transform)
    return (data, all_fitted_params)



@dataclass
class Manifest:
    split_config: tuple = (8, 1, 2)
    splitter: tuple | None = None
    pre_split_data_selector: tuple | None = None
    feature_transforms: list = field(default_factory=list)
    fitted_transforms: list = field(default_factory=list)
    target_column: str | None = None
    target_columns: list | None = None
    target_class_config: TargetClassConfig | None = None
    architecture_function: Callable | None = None
    architecture_params: dict = field(default_factory=dict)
    metrics_params: dict = field(default_factory=dict)
    as_numpy: bool = True

    def configuration(self):
        """Canonical pipeline configuration for experiment identity and provenance.

        Descriptors are data, so reading metadata never reconstructs fitting
        closures. The caller supplies the live manifest when resuming.
        """
        return encode(_configuration_value(self))

    def source_functions(self):
        """Live pipeline callables for verified caller-module snapshots."""
        functions, seen = [], set()
        def visit(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            if callable(value):
                functions.append(value)
                for name in ('func', 'args', 'keywords', '__defaults__', '__kwdefaults__'):
                    visit(getattr(value, name, None))
                for cell in getattr(value, '__closure__', None) or ():
                    visit(cell.cell_contents)
            elif is_dataclass(value):
                for item in fields(value):
                    visit(getattr(value, item.name))
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (tuple, list, set, frozenset)):
                for item in value:
                    visit(item)
        visit(self)
        return functions

    def add_transform(self, func, group=None, include_if=None, **params):
        self.feature_transforms.append(TransformEntry(func, dict(params), group, include_if))
        return self

    def add_fitted_transform(self, computations, func=None, **params):
        entry = computations if func is None else (computations, func, dict(params))
        if not isinstance(entry, tuple) or len(entry) != 3:
            raise TypeError('Fitted transform requires (computations, transform, params).')
        self.fitted_transforms.append(entry)
        return self

    def set_pre_split_data_selector(self, func, **params):
        self.pre_split_data_selector = (func, dict(params))
        return self

    def set_split_config(self, train, val, test):
        if isinstance(train, bool) or train <= 0 or val < 0 or test < 0:
            raise ValueError('Train ratio must be positive; validation/test ratios must be non-negative.')
        self.split_config = (train, val, test)
        self.splitter = None
        return self

    def set_splitter(self, func, **params):
        if not callable(func):
            raise TypeError('splitter must be callable.')
        self.splitter = (func, dict(params))
        return self

    def set_random_split(self, seed=None):
        return self.set_splitter(split_random, ratios=self.split_config, seed=seed)

    def set_split_dates(self, train_start, train_end, val_start, val_end, test_start, test_end, *, time_col='datetime'):
        return self.set_splitter(split_by_dates, train_start=train_start, train_end=train_end,
                                val_start=val_start, val_end=val_end, test_start=test_start,
                                test_end=test_end, time_col=time_col)

    def set_target_column(self, name):
        self.target_column = name
        self.target_columns = None
        self.target_class_config = None
        return self

    def set_target_columns(self, names):
        if not names:
            raise ValueError('At least one target column is required.')
        self.target_columns = list(names)
        self.target_column = self.target_columns[0] if len(names) == 1 else None
        self.target_class_config = None
        return self

    def with_target_label(self, target_name, target_class, fit_params=None, transform_params=None):
        self.target_column = target_name
        self.target_columns = None
        self.target_class_config = TargetClassConfig(target_class, dict(fit_params or {}), dict(transform_params or {}))
        return self

    def with_reference_architecture(self, architecture_function):
        if not callable(architecture_function):
            raise TypeError('architecture_function must be callable.')
        self.architecture_function = architecture_function
        return self

    def with_params_override(self, **overrides):
        manifest = copy.deepcopy(self)
        split_config = overrides.pop('split_config', None)
        if split_config is not None:
            if len(split_config) != 3:
                raise ValueError('split_config must contain train, validation and test ratios.')
            manifest.set_split_config(*split_config)
        manifest.architecture_params.update(overrides)
        return manifest

    def resolve_model_kwargs(self, round_params):
        if self.architecture_function is None:
            raise ValueError('Configure .with_reference_architecture(func) before run_model().')
        supplied = {**round_params, **_resolve_params(self.architecture_params, round_params)}
        signature = inspect.signature(self.architecture_function)
        kwargs = {}
        for name, parameter in signature.parameters.items():
            if name == 'data' or parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if name in supplied:
                kwargs[name] = supplied[name]
            elif name == 'prediction_calibration_config' and getattr(self, 'prediction_calibration_config', None) is not None:
                continue
            elif parameter.default is inspect.Parameter.empty:
                raise ValueError(f'Missing model parameter {name!r}.')
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            metadata = {'_id', '_trial_id', '_param_hash', '_round_index', '_injected', '_generation_index', '_search_strategy'}
            kwargs.update({name: value for name, value in supplied.items() if name not in metadata and name not in kwargs})
        return kwargs

    def _split_data(self, data, round_params):
        if isinstance(data, dict) and {'train', 'val', 'test'} <= set(data):
            splits = [data[name] for name in ('train', 'val', 'test')]
        elif isinstance(data, (list, tuple)):
            splits = list(data)
        elif isinstance(data, pl.DataFrame):
            if self.pre_split_data_selector is not None:
                function, params = self.pre_split_data_selector
                data = function(data, **_resolve_params(params, round_params))
            if self.splitter is None:
                splits = split_sequential(data, self.split_config)
            else:
                function, params = self.splitter
                splits = function(data, **_resolve_params(params, round_params))
        else:
            raise TypeError('Supply a Polars DataFrame or caller-owned [train, validation, test] tables.')
        if len(splits) != 3 or any(not isinstance(split, pl.DataFrame) for split in splits):
            raise TypeError('Splitter must return three Polars DataFrames.')
        if splits[0].height == 0:
            raise ValueError('Training split must contain observations.')
        return [split.clone() for split in splits]

    def prepare_data(self, data, round_params):
        return _prepare(self, data, round_params)

    def run_model(self, data, round_params):
        return self.architecture_function(data, **self.resolve_model_kwargs(round_params))


@dataclass
class MLManifest(Manifest):
    scaler: FittedTransformEntry | None = None
    prediction_calibration_config: CalibrationConfig | None = None
    ablation_config: AblationConfig | None = None
    pca_compression_config: PCACompressionConfig | None = None
    data_dict_extension: Callable | None = None
    strict_mode: bool = False

    def set_scaler(self, transform_class, param_name='_scaler', extra_params=None):
        self.scaler = make_fitted_scaler(param_name, transform_class, extra_params)
        return self

    def set_scaler_from_params(self, param_name='scaler_type', extra_params=None):
        static, dynamic = _split_extra_params(extra_params)
        def factory(data, scaler_type, **kwargs):
            if scaler_type not in SCALER_REGISTRY:
                raise ValueError(f'Unknown scaler {scaler_type!r}; choose {sorted(SCALER_REGISTRY)}.')
            return SCALER_REGISTRY[scaler_type](data, **static, **kwargs)
        self.scaler = ([('_scaler', factory, {'scaler_type': param_name, **dynamic})],
                       _apply_fitted_transform, {'fitted_transform': '_scaler'})
        return self

    def set_strict_mode(self, strict_mode):
        if not isinstance(strict_mode, bool):
            raise TypeError('strict_mode must be boolean.')
        self.strict_mode = strict_mode
        return self

    def set_feature_ablation(self, drop_count_key='feature_drop_count', seed_key='feature_drop_seed'):
        self.ablation_config = AblationConfig(drop_count_key, seed_key)
        return self

    def set_pca_compression(self, enabled_param='auto_pca', n_components_param='pca_k',
                            scaler_param_name='_scaler', component_prefix='pc_'):
        if any(not isinstance(value, str) or not value for value in
               (enabled_param, n_components_param, scaler_param_name, component_prefix)):
            raise TypeError('PCA parameter keys and component prefix must be non-empty strings.')
        self.pca_compression_config = PCACompressionConfig(enabled_param, n_components_param,
                                                          scaler_param_name, component_prefix)
        return self

    def add_to_data_dict(self, function):
        self.data_dict_extension = function
        return self

    def with_calibration(self):
        return CalibrationBuilder(self)

    def run_model(self, data, round_params):
        kwargs = self.resolve_model_kwargs(round_params)
        if self.prediction_calibration_config is not None:
            use_calibration = round_params.get('use_calibration', True)
            use_threshold = round_params.get('use_threshold', True)
            if not isinstance(use_calibration, bool) or not isinstance(use_threshold, bool):
                raise TypeError('Calibration and threshold flags must be boolean.')
            signature = inspect.signature(self.architecture_function)
            accepts_config = ('prediction_calibration_config' in signature.parameters or
                              any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()))
            if accepts_config:
                resolved = self.prediction_calibration_config.resolve(round_params)
                kwargs['prediction_calibration_config'] = CalibrationConfig(
                    resolved.calibration_func if use_calibration else None, resolved.calibration_params,
                    resolved.threshold_func if use_threshold else None, resolved.threshold_params)
            elif use_calibration or use_threshold:
                raise ValueError('Configured model must accept prediction_calibration_config or **kwargs.')
        return self.architecture_function(data, **kwargs)


MachineLearningManifest = MLManifest


def _configuration_value(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {'type': type(value), 'fields': {item.name: _configuration_value(getattr(value, item.name))
                                             for item in fields(value)}}
    if isinstance(value, dict):
        return {key: _configuration_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_configuration_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_configuration_value(item) for item in value)
    return value


def _prepare(manifest, raw_data, round_params):
    splits = manifest._split_data(raw_data, round_params)
    fitted = {}
    targets = list(manifest.target_columns or ([manifest.target_column] if manifest.target_column else []))
    drop_columns = None
    for index, split in enumerate(splits):
        training = index == 0
        data = split
        for entry in manifest.feature_transforms:
            if _should_include_transform(entry, round_params):
                data = entry.func(data, **_resolve_params(entry.params, round_params))
                if isinstance(data, pl.LazyFrame):
                    data = data.collect()
                if not isinstance(data, pl.DataFrame):
                    raise TypeError('Transforms must return a Polars DataFrame or LazyFrame.')
        if manifest.target_class_config is not None:
            data, fitted = _apply_class_based_target(manifest, data, round_params, fitted, training)
        data, fitted = _apply_fitted_transforms(manifest.fitted_transforms, data, round_params, fitted, training)
        if getattr(manifest, 'scaler', None) is not None:
            target_values = data.select(targets) if targets else None
            inputs = data.drop(targets) if targets else data
            inputs, fitted = _apply_fitted_transforms([manifest.scaler], inputs, round_params, fitted, training)
            data = inputs.hstack(target_values) if target_values is not None else inputs
        ablation = getattr(manifest, 'ablation_config', None)
        if ablation is not None:
            count = round_params.get(ablation.drop_count_key, 0)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError('Feature drop count must be a non-negative integer.')
            if drop_columns is None:
                candidates = [column for column in data.columns if column not in targets]
                if count >= len(candidates):
                    raise ValueError('Feature ablation must leave at least one input column.')
                seed = round_params.get(ablation.seed_key, 0)
                drop_columns = random.Random(seed).sample(sorted(candidates), count)
            data = data.drop(drop_columns)
        if getattr(manifest, 'strict_mode', False):
            numeric = [name for name, dtype in data.schema.items() if dtype.is_float()]
            if any(data.null_count().row(0)) or any(not data[name].is_finite().all() for name in numeric):
                raise ValueError(f'Split {index} contains null or non-finite values after preparation.')
        splits[index] = data
    if any(split.columns != splits[0].columns for split in splits):
        raise ValueError('Prepared split columns must match training columns and order.')
    splits, fitted = _compress_pca(manifest, splits, round_params, fitted, targets)
    if targets:
        output = split_data_to_prep_output(splits, target_cols=targets, as_numpy=manifest.as_numpy)
    else:
        output = {'x_' + name: split.to_numpy() if manifest.as_numpy else split
                  for name, split in zip(('train', 'val', 'test'), splits)}
    features = [column for column in splits[0].columns if column not in targets]
    output.update(fitted)
    output['_fitted_params'] = dict(fitted)
    output['_feature_names'] = features
    output['_target_names'] = targets
    output['_dropped_features'] = list(drop_columns or [])
    extension = getattr(manifest, 'data_dict_extension', None)
    if extension is not None:
        output = extension(data_dict=output, split_data=splits, round_params=round_params, fitted_params=fitted)
        if not isinstance(output, dict):
            raise TypeError('Data dictionary extension must return a dictionary.')
    return output


def _compress_pca(manifest, splits, round_params, fitted, targets):
    config = getattr(manifest, 'pca_compression_config', None)
    if config is None:
        return splits, fitted
    enabled = round_params.get(config.enabled_param, False)
    if not isinstance(enabled, bool):
        raise TypeError('PCA enable parameter must be boolean.')
    if not enabled:
        return splits, fitted
    if config.n_components_param not in round_params:
        raise ValueError(f'Missing PCA component parameter {config.n_components_param!r}.')
    count = round_params[config.n_components_param]
    features = [column for column in splits[0].columns if column not in targets]
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= min(len(features), splits[0].height):
        raise ValueError('PCA components must fit the training rows and input dimensions.')
    if any(not splits[0][column].dtype.is_numeric() for column in features):
        raise ValueError('PCA requires numeric input columns.')
    pca = PCA(n_components=count)
    pca.fit(splits[0].select(features).to_numpy())
    names = [config.component_prefix + str(index) for index in range(count)]
    transformed = []
    for split in splits:
        values = pca.transform(split.select(features).to_numpy()) if len(split) else np.empty((0, count))
        result = pl.DataFrame({name: values[:, index] for index, name in enumerate(names)})
        if targets:
            result = result.hstack(split.select(targets))
        transformed.append(result)
    fitted.update({'_pca': pca, '_pca_input_feature_names': features,
                   '_pca_feature_names': names, '_pca_n_components': count})
    return transformed, fitted


__all__ = ['Manifest', 'MLManifest', 'MachineLearningManifest', 'TransformEntry',
           'TargetClassConfig', 'CalibrationConfig', 'CalibrationBuilder', 'PCACompressionConfig',
           'AblationConfig', 'make_fitted_scaler']
