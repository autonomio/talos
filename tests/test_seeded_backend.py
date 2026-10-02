"""Declared lazy Torch backend receives its seed before its first model exists."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def test_declared_torch_seed_matches_fresh_runs_and_resumed_weights(tmp_path):
    if importlib.util.find_spec('torch') is None:
        pytest.skip('Torch backend is not installed')
    source = Path(__file__).resolve().parents[1] / 'examples/sfd/torch_sfd.py'
    outputs = []
    for name, pause in [('first', False), ('second', False), ('resumed', True)]:
        directory = tmp_path / name
        report = tmp_path / (name + '.json')
        script = f'''import json
from talos import run
from sklearn.datasets import load_iris
values={{'neurons':[4,8], 'learning_rate':[0.01], 'epochs':[1], 'batch_size':[16]}}
result=run({str(source)!r},params=values,backend='torch',seed=42,experiment_dir={str(directory)!r},stop_after={1 if pause else None!r},progress_bar=False)
'''
        if pause:
            script += f"result=run({str(source)!r},params=values,backend='torch',seed=42,experiment_dir={str(directory)!r},resume=True,progress_bar=False)\n"
        script += f"x,_=load_iris(return_X_y=True)\nopen({str(report)!r},'w').write(json.dumps({{'metrics':result.data[['loss','val_loss']].to_dict(orient='list'),'ids':result.data._trial_id.tolist(),'predictions':result.predict(x).tolist()}}))"
        completed = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        outputs.append(json.loads(report.read_text()))
    assert outputs[0]['ids'] == outputs[1]['ids'] == outputs[2]['ids']
    assert outputs[0]['metrics'] == outputs[1]['metrics'] == outputs[2]['metrics']
    np.testing.assert_array_equal(outputs[0]['predictions'], outputs[1]['predictions'])
    np.testing.assert_array_equal(outputs[0]['predictions'], outputs[2]['predictions'])
