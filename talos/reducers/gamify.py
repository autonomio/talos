from pathlib import Path
from .GamifyMap import GamifyMap


def gamify(self):
    """Apply legacy parameter status edits, including after checkpoint resume."""
    if not hasattr(self, '_gamify_object'):
        self._gamify_object = GamifyMap(self)
        if not Path(self._gamify_object._filename + '.json').exists():
            self._gamify_object.export_json()
            return self
    self._gamify_object.import_json()
    self._gamify_object.run_updates()
    self._gamify_object.export_json()
    return self
