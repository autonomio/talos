"""Read native Talos archives and historical JSON/H5 deploy packages."""
import json
import tempfile
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from talos.backends import backend_for


class Restore:
    def __init__(self, path_to_zip, custom_objects=None, model_factory=None):
        self.path = str(path_to_zip)
        self._temporary = tempfile.TemporaryDirectory(prefix='talos-restore-')
        folder = Path(self._temporary.name)
        self.run_dir = folder
        with ZipFile(path_to_zip) as archive:
            for member in archive.infolist():
                target = (folder / member.filename).resolve()
                if not target.is_relative_to(folder.resolve()):
                    raise ValueError('Archive member lies outside the restoration directory.')
            archive.extractall(folder)
        if (folder / 'manifest.json').exists():
            manifest = json.loads((folder / 'manifest.json').read_text())
            if manifest['talos_archive_version'] != 2:
                raise ValueError('Unsupported Talos archive version.')
            self.manifest = manifest
            from talos.experiment.source_snapshot import hydrate_sources
            hydrate_sources(manifest, folder)
            self.metadata = {'source_bundle': manifest.get('source_bundle')}
            self.parameter_columns = manifest.get('parameter_columns', {})
            descriptor = manifest['artifact']
            descriptor['path'] = str(folder / descriptor['path'])
            if descriptor.get('factory_source'):
                descriptor['factory_source'] = str(folder / descriptor['factory_source'])
            self.model = backend_for(backend=descriptor['backend']).load(descriptor, custom_objects, model_factory)
            self.results = pd.read_csv(folder / 'results.csv')
            self.details = pd.Series(json.loads((folder / 'details.json').read_text()))
            self.params = np.load(folder / 'params.npy', allow_pickle=True).item()
            self.round_history = json.loads((folder / 'history.json').read_text())
            selected = manifest['selected_model_id']
            self.artifacts = {selected: descriptor}
            self.models = {selected: self.model}
            self.x = np.load(folder / 'x.npy', allow_pickle=True).item()
            self.y = np.load(folder / 'y.npy', allow_pickle=True).item()
        else:
            models = list(folder.glob('*_model.json'))
            if len(models) != 1:
                raise ValueError('The legacy archive must contain exactly one model JSON.')
            prefix = str(models[0])[:-len('_model.json')]
            descriptor = {'backend': 'tensorflow', 'format': 'keras_json_weights',
                          'path': prefix + '_model.json', 'weights': prefix + '_model.h5'}
            self.model = backend_for(backend='tensorflow').load(descriptor, custom_objects)
            self.results = pd.read_csv(prefix + '_results.csv')
            self.results = self.results.loc[:, ~self.results.columns.str.startswith('Unnamed:')]
            self.details = pd.read_csv(prefix + '_details.txt', header=None)
            self.params = np.load(prefix + '_params.npy', allow_pickle=True).item()
            self.x = self._sample(prefix + '_x.csv')
            self.y = self._sample(prefix + '_y.csv')
            self.round_history = []
        self.data = self.results

    @staticmethod
    def _sample(path):
        if not Path(path).stat().st_size:
            return pd.DataFrame()
        return pd.read_csv(path, header=None)
