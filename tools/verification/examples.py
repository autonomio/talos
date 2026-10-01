"""Run every checked-in notebook, executable script and native SFD example."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    helpers = Path(__file__).resolve().parent
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(filter(None, (str(root), env.get('PYTHONPATH'))))
    records = []
    with tempfile.TemporaryDirectory(prefix='talos-examples-') as temporary:
        for index, path in enumerate(sorted((root/'examples').rglob('*'))):
            if path.suffix not in ('.py', '.ipynb'):
                continue
            record = {'path': str(path.relative_to(root)), 'whole_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            receipt = output.parent/f'example-{index}.json'
            log = output.parent/f'example-{index}.log'
            record['log'] = str(log)
            if path.suffix == '.ipynb':
                kind = 'notebook'
                command = [sys.executable, str(helpers/'notebooks.py'), str(path), str(receipt)]
            elif path.parent.name == 'sfd':
                kind = 'sfd'
                command = [sys.executable, str(helpers/'sfd_examples.py'), str(path), str(receipt)]
            else:
                kind = 'script'
                command = [sys.executable, str(path)]
            record['kind'] = kind
            started = time.monotonic()
            try:
                with log.open('w') as stream:
                    process = subprocess.run(command, cwd=temporary, env=env, stdout=stream,
                                             stderr=subprocess.STDOUT, timeout=300)
                record['status'] = 'passed' if process.returncode == 0 else 'failed'
                record['returncode'] = process.returncode
                if receipt.exists():
                    detail = json.loads(receipt.read_text())
                    record['evidence'] = detail
                    if detail.get('status') != 'passed':
                        record['status'] = 'failed'
            except subprocess.TimeoutExpired:
                record.update(status='failed', error='300-second timeout')
            record['seconds'] = round(time.monotonic()-started, 3)
            records.append(record)
            output.write_text(json.dumps({'examples': records}, indent=2)+'\n')
            print(record['path'], record['status'], flush=True)
    return int(any(record['status'] != 'passed' for record in records))


if __name__ == '__main__':
    raise SystemExit(main())
