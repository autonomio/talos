"""Verify the guarded Torch documentation factory restores without training or loading data."""
import argparse
import tempfile
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument('--output', type=Path, required=True)
arguments = parser.parse_args()
root = arguments.root.resolve()
output = arguments.output.resolve()
output.parent.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root))

import numpy as np
import talos
from talos.backends import backend_for

work=Path(tempfile.mkdtemp(prefix='guarded-torch-', dir=output.parent))
caller=work/'caller'
training=work/'training'
restore=work/'fresh-restore'
for directory in (caller,training,restore): directory.mkdir(parents=True,exist_ok=True)
page=root/'docs/Examples_PyTorch_Code.md'
code=re.search(r'```python\n(.*?)```',page.read_text(),re.S).group(1)
source=caller/'guarded_example.py'
source.write_text(code)
name='talos_docs_guarded_torch_archive'
spec=importlib.util.spec_from_file_location(name,source)
module=importlib.util.module_from_spec(spec)
sys.modules[name]=module
before=set(training.rglob('*'))
os.chdir(training)
spec.loader.exec_module(module)
assert not hasattr(module,'scan_object')
assert set(training.rglob('*'))==before, 'Import started an experiment'
scan=module.run_example()
assert len(scan.data)==2
model=scan.best_model('val_loss',asc=True)
x=scan.x_val.detach().cpu().numpy()
expected=backend_for(model).predict(model,x)
np.save(work/'x.npy',x)
np.save(work/'expected.npy',expected)
archive=talos.Deploy(scan,work/'guarded_model','val_loss',asc=True)
with zipfile.ZipFile(archive.path) as zipped:
 manifest=json.loads(zipped.read('manifest.json'))
 assert manifest['artifact']['factory'] is not None
 assert manifest.get('source_bundle')
shutil.rmtree(caller)
assert not source.exists()
child=r'''import json,numpy as np,talos,torch
from pathlib import Path
import sklearn.datasets
from talos.backends import backend_for
from talos.experiment import runner
calls=[]
def forbidden(*args,**kwargs):
    calls.append('unexpected-training-or-data-acquisition')
    raise AssertionError('Restoring the guarded factory retrained or loaded training data')
talos.Scan=forbidden
runner.run=forbidden
torch.optim.Adam.step=forbidden
sklearn.datasets.load_breast_cancer=forbidden
before=set(Path.cwd().rglob('*'))
restored=talos.Restore(ARCHIVE)
actual=backend_for(restored.model).predict(restored.model,np.load(INPUT))
np.testing.assert_allclose(actual,np.load(EXPECTED),rtol=1e-6,atol=1e-7)
np.testing.assert_array_equal(actual.argmax(axis=1),np.load(EXPECTED).argmax(axis=1))
assert not calls
assert set(Path.cwd().rglob('*'))==before
Path(RECEIPT).write_text(json.dumps({'status':'passed','fresh_process':True,'caller_source_deleted':True,
    'prediction_parity':True,'class_parity':True,'scan_and_optimizer_and_loader_forbidden':True,
    'no_retraining_on_import':True,'prediction_shape':list(actual.shape)},indent=2))
'''
for key,value in {'ARCHIVE':archive.path,'INPUT':str(work/'x.npy'),'EXPECTED':str(work/'expected.npy'),'RECEIPT':str(work/'restore.json')}.items():
 child=child.replace(key,repr(value))
result=subprocess.run([sys.executable,'-c',child],cwd=restore,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=os.environ.copy())
(work/'fresh_restore.log').write_text(result.stdout)
assert result.returncode==0, result.stdout
receipt=json.loads((work/'restore.json').read_text())
receipt.update({'page':'docs/Examples_PyTorch_Code.md','source_code_sha256':hashlib.sha256(code.encode()).hexdigest(),
                'archive':archive.path,'archive_sha256':hashlib.sha256(Path(archive.path).read_bytes()).hexdigest(),
                'log':str(work/'fresh_restore.log'),'script':__file__,'training_trials':len(scan.data)})
output.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
