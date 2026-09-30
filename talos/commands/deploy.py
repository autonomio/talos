"""Package a selected trained model and its experiment provenance locally."""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
from talos.backends import backend_for
from talos.utils.best_model import best_model, activate_model


def _json(value):
    if hasattr(value, 'item'):
        return value.item()
    if hasattr(value, 'tolist'):
        return value.tolist()
    return str(value)


def _copy_sources(bundle, run_dir, destination):
    run_dir = Path(run_dir).resolve()
    directory = (run_dir / bundle['directory']).resolve()
    if not directory.is_relative_to(run_dir):
        raise ValueError('Caller source snapshot directory lies outside run directory')
    verified = []
    for name, specification in bundle['modules'].items():
        path = (run_dir / specification['path']).resolve()
        if not path.is_relative_to(directory):
            raise ValueError('Caller source snapshot lies outside its module directory: ' + name)
        if specification.get('namespace'):
            if not path.is_dir():
                raise ValueError('Missing caller source snapshot: ' + name)
            verified.append((specification['path'], None))
        else:
            if not path.is_file():
                raise ValueError('Missing caller source snapshot: ' + name)
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest() != specification['sha256']:
                raise ValueError('Caller source snapshot checksum mismatch: ' + name)
            verified.append((specification['path'], content))
    for relative, content in verified:
        target = destination / relative
        if content is None:
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)


class Deploy:
    def __init__(self, scan_object, model_name, metric, asc=False, saved=False,
                 custom_objects=None, model_factory=None):
        self.scan_object = scan_object
        self.model_name = str(model_name)
        self.metric, self.asc = metric, asc
        self.data = scan_object.data
        model_factory = model_factory if model_factory is not None else getattr(scan_object, 'model_factory', None)
        self.best_model = best_model(scan_object, metric, asc)
        self.model = activate_model(scan_object, self.best_model, saved, custom_objects, model_factory)
        destination = Path(model_name)
        if destination.suffix == '.zip':
            destination = destination.with_suffix('')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='talos-deploy-') as temp:
            stage = Path(temp)
            source_bundle = getattr(scan_object, 'metadata', {}).get('source_bundle')
            if source_bundle:
                _copy_sources(source_bundle, scan_object.run_dir, stage)
            descriptor = backend_for(self.model).save(self.model, stage / 'model', model_factory)
            if model_factory is None and hasattr(scan_object, 'artifacts'):
                artifacts = scan_object.artifacts
                existing = artifacts.get(self.best_model) if isinstance(artifacts, dict) else artifacts[self.best_model]
                if existing and existing.get('factory'):
                    for key in ('factory', 'factory_source', 'factory_source_sha256', 'config'):
                        descriptor[key] = existing.get(key)
                    descriptor['factory_required'] = False
            if descriptor.get('factory_source'):
                factory_source = Path(descriptor['factory_source'])
                if factory_source.is_file():
                    shutil.copy2(factory_source, stage / 'factory.py')
                    descriptor['factory_source'] = 'factory.py'
            descriptor['path'] = Path(descriptor['path']).relative_to(stage).as_posix()
            manifest = {'talos_archive_version': 2, 'selected_model_id': int(self.best_model),
                        'artifact': descriptor, 'metric': metric, 'ascending': asc,
                        'parameter_columns': getattr(scan_object, 'parameter_columns', {})}
            if source_bundle:
                manifest['source_bundle'] = source_bundle
            (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=_json))
            scan_object.data.to_csv(stage / 'results.csv', index=False)
            details = scan_object.details.to_dict() if hasattr(scan_object.details, 'to_dict') else scan_object.details
            (stage / 'details.json').write_text(json.dumps(details, default=_json))
            # Retain arbitrary Python parameter values, as historical Talos archives do.
            np.save(stage / 'params.npy', scan_object.params, allow_pickle=True)
            for name in ('x', 'y'):
                value = getattr(scan_object, name, None)
                def sample(data):
                    if isinstance(data, dict):
                        return {key: sample(item) for key, item in data.items()}
                    if isinstance(data, list):
                        return [sample(item) for item in data]
                    if hasattr(data, 'detach'):
                        data = data.detach().cpu().numpy()
                    return data[:100] if data is not None else None
                box = np.empty((), dtype=object)
                box[()] = sample(value)
                np.save(stage / (name + '.npy'), box, allow_pickle=True)
            (stage / 'history.json').write_text(json.dumps(getattr(scan_object, 'round_history', []), default=_json))
            (stage / 'README.txt').write_text("Load this trusted Talos archive with talos.Restore(path).\n")
            self.path = shutil.make_archive(str(destination), 'zip', stage)

    def save_model_as(self):
        return self.path

    def save_details(self):
        return self.scan_object.details

    def save_data(self):
        return self.scan_object.x, self.scan_object.y

    def save_results(self):
        return self.scan_object.data

    def save_params(self):
        return self.scan_object.params

    def save_readme(self):
        return self.path

    def package(self):
        return self.path
