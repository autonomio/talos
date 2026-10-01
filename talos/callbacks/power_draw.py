import subprocess
import time
from .experiment_log import _callback_base


class PowerDraw:
    def __new__(cls, device=0, backend='keras', provider=None):
        base = _callback_base(backend)
        def measure():
            if provider is not None:
                return float(provider())
            result = subprocess.run(['nvidia-smi', '-i', str(device), '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                                    check=True, capture_output=True, text=True)
            return float(result.stdout.strip())
        class PowerCallback(base):
            def __init__(self):
                super().__init__()
                self.log = {'epoch_begin': [], 'epoch_end': [], 'seconds': []}

            def on_train_begin(self, logs=None):
                self.log = {'epoch_begin': [], 'epoch_end': [], 'seconds': []}

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch_start_time = time.monotonic()
                self.log['epoch_begin'].append(measure())

            def on_epoch_end(self, epoch, logs=None):
                self.log['epoch_end'].append(measure())
                self.log['seconds'].append(time.monotonic() - self.epoch_start_time)
        return PowerCallback()
