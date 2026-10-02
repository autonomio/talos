"""Framework callback factory for explicit trial-level epoch logs."""
import csv
import json
import uuid
from pathlib import Path

from talos.experiment.context import get_trial_context


def _callback_base(backend):
    if backend in ('torch', 'pytorch', 'generic'):
        return object
    if backend in ('tensorflow', 'tf', 'tf.keras'):
        from tensorflow.keras.callbacks import Callback
    else:
        from keras.callbacks import Callback
    return Callback


class ExperimentLog:
    def __new__(cls, experiment_name, params, backend='keras'):
        base = _callback_base(backend)
        context = get_trial_context() or {}
        folder = Path(context.get('run_dir', experiment_name))
        folder.mkdir(parents=True, exist_ok=True)
        trial = str(context.get('trial_id', uuid.uuid4().hex))

        class EpochLog(base):
            def __init__(self):
                super().__init__()
                self.name = str(folder / ('epochs-' + trial + '.log'))
                self.params = params
                self.counter = 1

            def on_train_begin(self, logs=None):
                self.final_out = []
                self.hash = trial
                self.keys = None

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                if self.keys is None:
                    self.keys = list(logs)
                self.final_out.append({'id': trial, 'epoch': epoch + 1, **logs})
                with open(self.name, 'a', newline='') as stream:
                    writer = csv.writer(stream)
                    if epoch == 0:
                        writer.writerow(['id', 'epoch', *self.keys, 'params'])
                    writer.writerow([trial, epoch + 1, *[logs.get(key) for key in self.keys], json.dumps(params, default=str, sort_keys=True)])

            def on_train_end(self, logs=None):
                return self.name
        return EpochLog()
