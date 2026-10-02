import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from talos import Deploy
from talos.backends import backend_for
from talos.experiment import run

source = Path(sys.argv[1]).resolve()
report_path = Path(sys.argv[2])
with tempfile.TemporaryDirectory(prefix='talos-sfd-example-') as temporary:
    folder = Path(temporary)
    caller = folder / source.name
    caller.write_bytes(source.read_bytes())
    result = run(caller, seed=17, progress_bar=False, experiment_dir=folder / 'run')
    assert len(result.data) == 4
    from talos.experiment.runner import load_sfd
    sfd = load_sfd(caller)
    data = sfd.prep(None, result._records[0]['params'])
    x = data['x_val']
    np.save(folder / 'x.npy', x)
    expected = result.predict(x, metric='val_loss', asc=True)
    archive = Deploy(result, folder / 'archive', metric='val_loss', asc=True)
    caller.unlink()
    script = ('import numpy as np; from talos import Restore; from talos.backends import backend_for; '
            f'restored=Restore({archive.path!r}); '
            f'prediction=backend_for(restored.model).predict(restored.model,np.load({str(folder / "x.npy")!r})); '
            f'np.save({str(folder / "restored.npy")!r},prediction)')
    fresh = subprocess.run([sys.executable, '-c', script], env=os.environ.copy(), capture_output=True, text=True)
    assert fresh.returncode == 0, fresh.stdout + fresh.stderr
    np.testing.assert_allclose(expected, np.load(folder / 'restored.npy'), rtol=1e-5, atol=1e-6)
    report = {'file': str(source), 'whole_file_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'code_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'status': 'passed',
            'trials': len(result.data), 'trial_ids': result.data['_trial_id'].tolist(),
            'dataset': 'sklearn bundled Iris', 'prediction_shape': list(np.asarray(expected).shape),
            'archive_sha256': hashlib.sha256(Path(archive.path).read_bytes()).hexdigest(),
            'fresh_process_restore_after_source_deletion': True,
            'artifacts': [{'backend': a['backend'], 'sha256': a['sha256']} for a in result.artifacts]}
    report_path.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
