"""Bounded regressions for the legacy security candidate, with real Iris behavior."""
import contextlib
import hashlib
import importlib
import io
import json
import tarfile
import zipfile
from importlib import metadata
from pathlib import Path

import pytest

installed_keras = next(metadata.distributions(name='keras'), None)
if installed_keras is None or installed_keras.version != '2.14.0+autonomio.1':
    pytest.skip('Owned Keras 2.14 security regressions run in the legacy CI lane', allow_module_level=True)

keras = importlib.import_module('keras')
h5py = importlib.import_module('h5py')
np = importlib.import_module('numpy')
any_pb2 = importlib.import_module('google.protobuf.any_pb2')
json_format = importlib.import_module('google.protobuf.json_format')
serialization_lib = importlib.import_module('keras.src.saving.serialization_lib')
saving_lib = importlib.import_module('keras.src.saving.saving_lib')
data_utils = importlib.import_module('keras.src.utils.data_utils')


def test_protobuf_any_depth_is_enforced():
    value = {}
    for _ in range(8):
        value = {'@type': 'type.googleapis.com/google.protobuf.Any', 'value': value}
    with pytest.raises(json_format.ParseError, match='Max recursion depth'):
        json_format.ParseDict(value, any_pb2.Any(), max_recursion_depth=5)
    json_format.ParseDict(value, any_pb2.Any(), max_recursion_depth=16)


def test_lambda_safe_default_and_explicit_unsafe_compatibility():
    config = keras.layers.Lambda(lambda x: x + 1).get_config()
    with pytest.raises(ValueError, match='Lambda'):
        keras.layers.Lambda.from_config(dict(config))
    with serialization_lib.SafeModeScope(False):
        restored = keras.layers.Lambda.from_config(dict(config))
        np.testing.assert_array_equal(restored(np.array([1.0])).numpy(), [2.0])


@pytest.mark.parametrize('module', ['keras.__internal__', 'keras.src.saving.serialization_lib'])
def test_serialized_config_cannot_disable_safe_mode(module):
    descriptor = {'class_name': 'function', 'module': module, 'config': 'enable_unsafe_deserialization', 'registered_name': None}
    with pytest.raises(ValueError, match='Non-modeling'):
        serialization_lib.deserialize_keras_object(descriptor)
    assert serialization_lib.in_safe_mode() is not False


def test_external_vocabulary_rejected_in_safe_mode(tmp_path):
    source = tmp_path / 'private.txt'
    source.write_text('private-token\n')
    config = keras.layers.StringLookup(vocabulary=['a']).get_config()
    config['vocabulary'] = str(source)
    with serialization_lib.SafeModeScope(True), pytest.raises(TypeError, match='external vocabulary'):
        keras.layers.StringLookup.from_config(config)


def test_vocabulary_asset_roundtrip_after_original_deleted(tmp_path):
    source = tmp_path / 'vocabulary.txt'
    source.write_text('a\nb\nc\n')
    model = keras.Sequential([keras.layers.Input(shape=(1,), dtype='string'), keras.layers.StringLookup(vocabulary=str(source))])
    expected = model(np.array([['a'], ['c']])).numpy()
    artifact = tmp_path / 'lookup.keras'
    model.save(artifact)
    source.unlink()
    restored = keras.models.load_model(artifact, compile=False)
    np.testing.assert_array_equal(restored(np.array([['a'], ['c']])).numpy(), expected)


@pytest.mark.parametrize('kind', ['external_link', 'soft_link', 'virtual', 'external_storage', 'shape_bomb'])
def test_hdf5_unsafe_storage_rejected_before_read(tmp_path, kind):
    external = tmp_path / 'private.h5'
    with h5py.File(external, 'w') as handle:
        handle.create_dataset('private', data=np.array([1., 2., 3.]))
    target = tmp_path / 'weights.h5'
    with h5py.File(target, 'w', libver='latest') as handle:
        group = handle.create_group('vars')
        if kind == 'external_link':
            group['0'] = h5py.ExternalLink(str(external), '/private')
        elif kind == 'soft_link':
            handle.create_dataset('private', data=[1., 2., 3.])
            group['0'] = h5py.SoftLink('/private')
        elif kind == 'virtual':
            layout = h5py.VirtualLayout(shape=(3,), dtype='f8')
            layout[:] = h5py.VirtualSource(str(external), 'private', shape=(3,))
            group.create_virtual_dataset('0', layout)
        elif kind == 'external_storage':
            group.create_dataset('0', shape=(3,), dtype='f8', external=[(str(tmp_path / 'secret.bin'), 0, h5py.h5f.UNLIMITED)])
        else:
            group.create_dataset('0', shape=(1 << 31,), chunks=(1,), dtype='f8')
    with pytest.raises(ValueError, match='HDF5'):
        store = saving_lib.H5IOStore(str(target), mode='r')
        store.close()


@pytest.mark.parametrize('path', ['../escape', '..\\escape', '/absolute', 'x/../../escape'])
def test_asset_paths_reject_escape(tmp_path, path):
    if path == '/absolute':
        path = str(tmp_path / 'outside-absolute')
    with contextlib.closing(saving_lib.DiskIOStore(str(tmp_path / 'assets'), mode='w')) as store:
        with pytest.raises(ValueError, match='Asset path'):
            store.make(path)
        with pytest.raises(ValueError, match='Asset path'):
            store.get(path)
        assert Path(store.make('layers/dense')).is_dir()


def test_tar_traversal_rejected_without_deleting_existing_files(tmp_path):
    archive = tmp_path / 'input.tar'
    output = tmp_path / 'output'
    output.mkdir()
    sentinel = output / 'existing.txt'
    sentinel.write_text('keep')
    with tarfile.open(archive, 'w') as handle:
        entry = tarfile.TarInfo('../outside.txt')
        entry.size = 4
        handle.addfile(entry, io.BytesIO(b'test'))
    with pytest.raises(tarfile.TarError):
        data_utils._extract_archive(str(archive), str(output), 'tar')
    assert sentinel.read_text() == 'keep'
    assert not (tmp_path / 'outside.txt').exists()


def test_zip_traversal_rejected(tmp_path):
    artifact = tmp_path / 'input.zip'
    with zipfile.ZipFile(artifact, 'w') as handle:
        handle.writestr('../outside.txt', 'test')
    with pytest.raises(ValueError, match='ZIP member'):
        data_utils._extract_archive(str(artifact), str(tmp_path / 'output'), 'zip')
    assert not (tmp_path / 'outside.txt').exists()


def test_genuine_keras214_archive_predictions_preserved():
    root = Path(__file__).resolve().parent / 'fixtures/keras214'
    provenance = json.loads((root / 'provenance.json').read_text())
    for item in provenance['files']:
        assert hashlib.sha256((root / item['path']).read_bytes()).hexdigest() == item['sha256']
    expected = np.load(root / 'expected.npz', allow_pickle=False)
    model = keras.models.load_model(root / 'genuine-keras214-native.keras', compile=False)
    x = expected['x']
    predictions = expected['prediction']
    np.testing.assert_allclose(model.predict(x, verbose=0), predictions, rtol=1e-5, atol=1e-6)


def test_tar_sibling_prefix_escape_rejected(tmp_path, monkeypatch):
    output = tmp_path / 'output'
    output.mkdir()
    monkeypatch.chdir(output)
    outside = tmp_path / 'output-sibling' / 'payload.txt'
    archive = tmp_path / 'prefix.tar'
    with tarfile.open(archive, 'w') as handle:
        entry = tarfile.TarInfo(str(outside))
        entry.size = 4
        handle.addfile(entry, io.BytesIO(b'test'))
    with pytest.raises(tarfile.TarError):
        data_utils._extract_archive(str(archive), str(output), 'tar')
    assert not outside.exists()


def test_zip_existing_directory_symlink_cannot_escape(tmp_path):
    output = tmp_path / 'output'
    outside = tmp_path / 'outside'
    output.mkdir()
    outside.mkdir()
    (output / 'pivot').symlink_to(outside, target_is_directory=True)
    archive = tmp_path / 'link.zip'
    with zipfile.ZipFile(archive, 'w') as handle:
        handle.writestr('pivot/payload.txt', 'test')
    with pytest.raises(ValueError, match='Asset path'):
        data_utils._extract_archive(str(archive), str(output), 'zip')
    assert not (outside / 'payload.txt').exists()


def test_npz_object_arrays_require_explicit_unsafe_scope(tmp_path):
    path = tmp_path / 'objects.npz'
    np.savez(path, layer=np.array({'0': np.array([1., 2.])}, dtype=object))
    with serialization_lib.SafeModeScope(True):
        store = saving_lib.NpzIOStore(str(path), mode='r')
        with contextlib.closing(store.f):
            with pytest.raises(ValueError, match='Object arrays'):
                store.get('layer')
    with serialization_lib.SafeModeScope(False):
        store = saving_lib.NpzIOStore(str(path), mode='r')
        with contextlib.closing(store.f):
            np.testing.assert_array_equal(store.get('layer')['0'], [1., 2.])


@pytest.mark.parametrize('kind', ['functional', 'lstm', 'registered_custom'])
@pytest.mark.parametrize('optimizer_kind', ['current', 'legacy'])
def test_model_and_optimizer_continuation_preserved(tmp_path, kind, optimizer_kind):
    import tensorflow as tf
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    keras.utils.set_random_seed(17)
    x, y = load_iris(return_X_y=True)
    x = StandardScaler().fit_transform(x).astype('float32')
    if kind == 'functional':
        inputs = keras.Input(shape=(4,))
        outputs = keras.layers.Dense(3, activation='softmax')(tf.math.square(inputs))
        model = keras.Model(inputs, outputs)
    elif kind == 'lstm':
        # Each real Iris observation is one timestep; no temporal inference claim.
        x = x.reshape(len(x), 1, 4)
        model = keras.Sequential([keras.layers.Input(shape=(1, 4)), keras.layers.LSTM(4), keras.layers.Dense(3, activation='softmax')])
    else:
        @keras.saving.register_keras_serializable(package='TalosCompatibility')
        class Scale(keras.layers.Layer):
            def call(self, inputs):
                return inputs * 2
        model = keras.Sequential([keras.layers.Input(shape=(4,)), Scale(), keras.layers.Dense(3, activation='softmax')])
    optimizer = keras.optimizers.legacy.Adam(lr=.01, decay=.001) if optimizer_kind == 'legacy' else keras.optimizers.Adam(learning_rate=.01)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')
    model.train_on_batch(x, y)
    path = tmp_path / (kind + ('.h5' if optimizer_kind == 'legacy' else '.keras'))
    model.save(path)
    restored = keras.models.load_model(path, compile=True)
    np.testing.assert_allclose(restored.predict(x, verbose=0), model.predict(x, verbose=0), rtol=1e-5, atol=1e-6)
    assert int(restored.optimizer.iterations) == int(model.optimizer.iterations)
    original_loss = model.train_on_batch(x, y)
    restored_loss = restored.train_on_batch(x, y)
    assert restored_loss == pytest.approx(original_loss, rel=1e-6, abs=1e-6)
    np.testing.assert_allclose(restored.predict(x, verbose=0), model.predict(x, verbose=0), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('kind', [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_tar_links_rejected_before_any_extraction(tmp_path, kind):
    archive = tmp_path / 'links.tar'
    output = tmp_path / 'output'
    with tarfile.open(archive, 'w') as handle:
        entry = tarfile.TarInfo('pivot')
        entry.type = kind
        entry.linkname = '../outside'
        handle.addfile(entry)
    with pytest.raises(tarfile.FilterError, match='links are prohibited'):
        data_utils._extract_archive(str(archive), str(output), 'tar')
    assert not (output / 'pivot').exists()


@pytest.mark.parametrize('module,name', [
    ('builtins', 'eval'), ('builtins', 'exec'), ('builtins', '__import__'),
    ('keras.models', 'load_model'),
    ('keras.src.layers.core.lambda_layer', 'generic_utils.func_load'),
])
def test_serialized_function_reexports_cannot_bypass_safe_mode(module, name):
    descriptor = {'class_name': 'function', 'module': module, 'config': name,
                  'registered_name': None}
    with pytest.raises(ValueError, match='Non-modeling function'):
        serialization_lib.deserialize_keras_object(descriptor)


def test_builtin_activation_and_explicit_custom_function_remain_available():
    descriptor = {'class_name': 'function', 'module': 'builtins', 'config': 'relu',
                  'registered_name': None}
    assert serialization_lib.deserialize_keras_object(descriptor) is keras.activations.relu
    descriptor['config'] = 'custom_scale'

    def custom_scale(inputs):
        return inputs * 2
    restored = serialization_lib.deserialize_keras_object(descriptor,
                    custom_objects={'custom_scale': custom_scale})
    assert restored is custom_scale
    assert restored(3) == 6


@pytest.mark.parametrize('malicious', [False, True])
def test_public_get_file_download_extraction_uses_repaired_boundary(tmp_path, malicious):
    archive = tmp_path / 'source.tar'
    with tarfile.open(archive, 'w') as handle:
        entry = tarfile.TarInfo('../outside.txt' if malicious else 'files/a.txt')
        entry.size = 4
        handle.addfile(entry, io.BytesIO(b'test'))
    arguments = {'fname': 'download.tar', 'origin': archive.as_uri(),
                 'cache_dir': str(tmp_path / 'cache'), 'extract': True,
                 'file_hash': hashlib.sha256(archive.read_bytes()).hexdigest()}
    if malicious:
        with pytest.raises(tarfile.TarError):
            keras.utils.get_file(**arguments)
        assert not (tmp_path / 'cache/outside.txt').exists()
    else:
        path = Path(keras.utils.get_file(**arguments))
        assert path.is_file()
        assert (path.parent / 'files/a.txt').read_bytes() == b'test'


def test_installed_backports_bind_complete_source_and_import_identity():
    from importlib import metadata

    import google.protobuf

    from tools.security.installed import verified_backport
    assert verified_backport('keras') == '2.14.0'
    assert verified_backport('protobuf') == '4.25.9'
    assert keras.__version__ == metadata.version('keras') == '2.14.0+autonomio.1'
    assert google.protobuf.__version__ == metadata.version('protobuf') == '4.25.9+autonomio.1'
    assert Path(keras.__file__).resolve() == Path(metadata.distribution('keras').locate_file('keras/__init__.py')).resolve()
    assert Path(google.protobuf.__file__).resolve() == Path(metadata.distribution('protobuf').locate_file('google/protobuf/__init__.py')).resolve()


def test_forged_record_cannot_hide_unmodified_upstream_source_drift(tmp_path, monkeypatch):
    import base64
    import csv
    import shutil
    from importlib import metadata

    from tools.security import installed
    real = metadata.distribution('keras')
    info = 'keras-2.14.0+autonomio.1.dist-info'
    for directory in ['keras', info]:
        shutil.copytree(real.locate_file(directory), tmp_path / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    file = 'keras/src/layers/core/activation.py'
    source = tmp_path / file
    source.write_bytes(source.read_bytes() + b'\n# changed upstream module\n')
    record = tmp_path / info / 'RECORD'
    rows = list(csv.reader(io.StringIO(record.read_text())))
    for row in rows:
        if row[0] == file:
            row[1] = 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(source.read_bytes()).digest()).decode().rstrip('=')
            row[2] = str(source.stat().st_size)
    with record.open('w', newline='') as output:
        csv.writer(output, lineterminator='\n').writerows(rows)
    cloned = metadata.PathDistribution(tmp_path / info)
    monkeypatch.setattr(installed.metadata, 'distribution', lambda name: cloned)
    with pytest.raises(ValueError, match='complete source inventory differs'):
        installed.verified_backport('keras')


def test_native_lambda_archive_retains_explicit_trusted_scope(tmp_path):
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    x, _ = load_iris(return_X_y=True)
    x = StandardScaler().fit_transform(x).astype('float32')
    model = keras.Sequential([keras.Input(shape=(4,)),
            keras.layers.Lambda(lambda inputs: inputs * 2), keras.layers.Dense(3)])
    expected = model.predict(x, verbose=0)
    artifact = tmp_path / 'trusted-lambda.keras'
    model.save(artifact)
    with pytest.raises(ValueError, match='Lambda'):
        keras.models.load_model(artifact, compile=False, safe_mode=None)
    with serialization_lib.SafeModeScope(False):
        restored = keras.models.load_model(artifact, compile=False)
        np.testing.assert_allclose(restored.predict(x, verbose=0), expected, rtol=1e-5, atol=1e-6)
    restored = keras.models.load_model(artifact, compile=False, safe_mode=False)
    np.testing.assert_allclose(restored.predict(x, verbose=0), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('name', ['keras', 'Keras'])
def test_owned_audit_lookup_retains_upstream_identity_for_normalized_names(name):
    from packaging.requirements import Requirement

    from tools.security.legacy_audit import legacy_identity
    from tools.security.wheels import MANIFEST
    entry = json.loads(MANIFEST.read_text())[0]
    requirement = Requirement(name + '==' + entry['patched_version'])
    assert legacy_identity(requirement, {entry['output_sha256']}) == (entry['patched_version'], entry['upstream_version'])


@pytest.mark.parametrize('mutation', ['missing-hash', 'extra-hash', 'wrong-hash', 'wrong-version'])
def test_owned_audit_lookup_rejects_altered_lock_binding(mutation):
    from packaging.requirements import Requirement

    from tools.security.legacy_audit import legacy_identity
    from tools.security.wheels import MANIFEST
    entry = json.loads(MANIFEST.read_text())[0]
    hashes = {entry['output_sha256']}
    version = entry['patched_version']
    if mutation == 'missing-hash':
        hashes.clear()
    elif mutation == 'extra-hash':
        hashes.add('0' * 64)
    elif mutation == 'wrong-hash':
        hashes = {'0' * 64}
    else:
        version = entry['upstream_version']
    with pytest.raises(ValueError, match='upstream lock entry'):
        legacy_identity(Requirement('keras==' + version), hashes)
