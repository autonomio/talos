"""Execute all assigned Markdown Python blocks in page order, with no training stubs."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get('TALOS_DOCS_ROOT', Path(__file__).resolve().parents[2]))
WORK = Path(os.environ.get('TALOS_DOCS_WORK', '/tmp/talos-docs-api-execution'))
MANIFEST = Path(os.environ.get('TALOS_DOCS_MANIFEST', '/tmp/talos-docs-examples-api.json'))
NAMES = ['AutoModel','AutoParams','AutoPredict','AutoScan','Generator','Hidden_Layers',
         'Learning_Rate_Normalizer','Metrics','Templates']
PAGES = sorted((ROOT/'docs').glob('Examples_*.md')) + [ROOT/'docs'/(name+'.md') for name in NAMES]

def extract(path):
    source = path.read_text()
    result = []
    for match in re.finditer(r'^```python[^\n]*\n(.*?)^```[ \t]*$', source, re.M|re.S):
        code = match.group(1)
        if not code.endswith('\n'):
            code += '\n'
        result.append({'path': str(path.relative_to(ROOT)),
                       'start_line': source.count('\n', 0, match.start()) + 1,
                       'code_sha256': hashlib.sha256(code.encode()).hexdigest(),
                       'code': code})
    return result

def child(page):
    import contextlib
    import traceback
    import types
    blocks = extract(page)
    directory = WORK/page.stem
    directory.mkdir(parents=True, exist_ok=True)
    os.chdir(directory)
    module_name = 'talos_docs_' + page.stem.lower()
    code_file = directory/(module_name+'.py')
    combined, offsets = '', []
    for block in blocks:
        offsets.append(combined.count('\n'))
        combined += block['code']+'\n'
    code_file.write_text(combined)
    module = types.ModuleType(module_name)
    module.__file__ = str(code_file)
    sys.modules[module_name] = module
    sys.path.insert(0, str(directory))
    sys.path.insert(0, str(ROOT))
    module.__dict__['__builtins__'] = __builtins__
    records = []
    for index, (block, offset) in enumerate(zip(blocks, offsets)):
        log = directory/f'block_{index + 1:03}.log'
        start = time.monotonic()
        record = dict(block)
        record.update({'context': {'mode': 'sequential_page', 'module': module_name,
                                  'working_directory': str(directory), 'prior_blocks': index,
                                  'fixtures': 'None; all imports, real data and training are authored on the page.'},
                       'runtime': {'python': sys.version.split()[0], 'executable': sys.executable,
                                   'keras_backend': os.environ.get('KERAS_BACKEND'), 'device': 'cpu'},
                       'log': str(log), 'execution_source': str(code_file)})
        with log.open('w') as output, contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                exec(compile('\n'*offset+block['code'], str(code_file), 'exec'), module.__dict__)
                if page.name == 'Examples_PyTorch_Code.md':
                    # Preserve an importable factory module while exercising the actual guarded recipe.
                    module.__dict__['scan_object'] = module.__dict__['run_example']()
                    record['context']['entrypoint'] = 'run_example() after importable definitions'
                    record['context']['training'] = 'Actual authored two-trial Torch optimizer training'
                record['status'] = 'passed'
            except BaseException as exc:
                record['status'] = 'failed'
                record['error'] = type(exc).__name__ + ': ' + str(exc)
                traceback.print_exc()
        record['duration_seconds'] = round(time.monotonic()-start, 4)
        records.append(record)
        (directory/'results.json').write_text(json.dumps(records, indent=2))
    print(json.dumps({'page': str(page.relative_to(ROOT)), 'passed': sum(r['status']=='passed' for r in records),
                      'failed': [dict(path=r['path'], start_line=r['start_line'], error=r['error'], log=r['log'])
                                 for r in records if r['status']!='passed']}))
    return any(r['status']!='passed' for r in records)

def parent(selected):
    WORK.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({'PYTHONPATH': os.pathsep.join(filter(None, (str(ROOT), os.environ.get('PYTHONPATH')))), 'KERAS_BACKEND': 'tensorflow', 'KERAS_TORCH_DEVICE': 'cpu',
                'CUDA_VISIBLE_DEVICES': '-1', 'TF_NUM_INTRAOP_THREADS': '1', 'TF_NUM_INTEROP_THREADS': '1',
                'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MPLBACKEND': 'Agg'})
    pages = [p for p in PAGES if not selected or p.stem in selected]
    child_failures = []
    for page in pages:
        result = WORK/page.stem/'results.json'
        if result.exists():
            result.unlink()
        process = subprocess.run([sys.executable, __file__, '--child', str(page)], env=env,
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        print(process.stdout, flush=True)
        if process.returncode:
            child_failures.append({'page': str(page.relative_to(ROOT)), 'returncode': process.returncode})
    archive_check = None
    if any(page.name == 'Examples_PyTorch_Code.md' for page in pages):
        archive_receipt = WORK/'guarded-archive.json'
        process = subprocess.run([sys.executable, str(Path(__file__).with_name('torch_docs_archive.py')),
                                  '--root', str(ROOT), '--output', str(archive_receipt)],
                                 env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        (WORK/'guarded-archive.log').write_text(process.stdout)
        if process.returncode:
            child_failures.append({'page': 'Examples_PyTorch_Code.md archive', 'returncode': process.returncode})
        elif archive_receipt.exists():
            archive_check = json.loads(archive_receipt.read_text())
    records = []
    for page in PAGES:
        result = WORK/page.stem/'results.json'
        if result.exists():
            records.extend(json.loads(result.read_text()))
    versions = {}
    for name in ('tensorflow', 'keras', 'torch', 'numpy', 'scikit-learn'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    for record in records:
        record['runtime']['versions'] = versions
    expected = {(block['path'], block['start_line']): block['code_sha256']
                for page in PAGES for block in extract(page)}
    actual = {(block['path'], block['start_line']): block['code_sha256'] for block in records}
    inventory_verified = expected == actual
    payload = {'schema_version': 1, 'owner': '/root/talos_api_contract',
               'execution_script': __file__, 'root': str(ROOT),
               'environment': sys.prefix, 'real_datasets': ['Iris','Wisconsin Breast Cancer','Handwritten Digits'],
               'inventory_verified': inventory_verified,
               'executed_inventory_verified': all(expected.get(key) == value for key, value in actual.items()),
               'unexecuted_blocks': [{'path': key[0], 'start_line': key[1], 'code_sha256': value}
                                     for key, value in expected.items() if key not in actual],
               'full_code_pages': [{'path': str(page.relative_to(ROOT)), 'blocks': len(extract(page)),
                                    'status': ('not_executed' if not any(r['path']==str(page.relative_to(ROOT)) for r in records)
                                               else 'passed' if all(r['status']=='passed' for r in records if r['path']==str(page.relative_to(ROOT))) else 'failed'),
                                    'training': 'Actual authored framework training; no substitutes.'}
                                   for page in PAGES if page.stem.endswith('_Code')],
               'portable_archive_check': archive_check, 'child_failures': child_failures, 'total_blocks': len(records), 'passed': sum(r['status']=='passed' for r in records),
               'failed': sum(r['status']!='passed' for r in records), 'blocks': records}
    MANIFEST.write_text(json.dumps(payload, indent=2))
    print(json.dumps({key:payload[key] for key in ('total_blocks','passed','failed')}), flush=True)
    requested = {(block['path'], block['start_line']): block['code_sha256']
                 for page in pages for block in extract(page)}
    return bool(payload['failed'] or child_failures) or any(actual.get(key) != value for key, value in requested.items())

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', type=Path)
    parser.add_argument('pages', nargs='*')
    arguments = parser.parse_args()
    sys.exit(child(arguments.child) if arguments.child else parent(arguments.pages))
