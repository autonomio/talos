#!/usr/bin/env python3
"""Reuse the exact successful documentation test command for identical source and dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

COMMAND = ['-m', 'coverage', 'run', '-m', 'pytest', '-q']
ENVIRONMENT = ('KERAS_BACKEND', 'KERAS_TORCH_DEVICE', 'CUDA_VISIBLE_DEVICES',
               'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS', 'OMP_NUM_THREADS',
               'OPENBLAS_NUM_THREADS', 'MPLBACKEND')


def environment(root: Path, python: str) -> dict[str, object]:
    """Compare the interpreter and every declared documentation dependency."""
    lock = (root / 'requirements/ci/documentation-3.12.txt').read_text(encoding='utf-8')
    names = sorted(re.findall(r'^([A-Za-z0-9_.-]+)==', lock, re.MULTILINE))
    script = ('import importlib.metadata as m,json,sys; '
              'print(json.dumps({"python":sys.version,"dependencies":'
              '{name:m.version(name) for name in json.loads(sys.argv[1])}}))')
    result = json.loads(subprocess.check_output([python, '-c', script, json.dumps(names)], text=True))
    result['environment'] = {name: os.environ.get(name) for name in ENVIRONMENT}
    return result


def record(root: Path, output: Path) -> None:
    """Record only after the workflow's exact full-suite command succeeds."""
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    files = {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
             for path in paths if path and (root / path).is_file()}
    receipt = {'command': COMMAND, 'files': files, 'runtime': environment(root, sys.executable),
               'source': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
               'run_id': os.environ.get('GITHUB_RUN_ID'),
               'run_attempt': os.environ.get('GITHUB_RUN_ATTEMPT'), 'returncode': 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')


def verify(root: Path, python: str, receipt: Path) -> dict[str, object]:
    """Missing, changed or unsuccessful evidence fails; it never skips verification."""
    proof = json.loads(receipt.read_text(encoding='utf-8'))
    if proof['command'] != COMMAND or proof['returncode'] != 0:
        raise ValueError('development evidence is not the successful exact test command')
    if proof['run_id'] != os.environ.get('GITHUB_RUN_ID'):
        raise ValueError('development evidence belongs to another workflow run')
    if proof.get('run_attempt') != os.environ.get('GITHUB_RUN_ATTEMPT'):
        raise ValueError('development evidence belongs to another workflow attempt')
    if os.environ.get('GITHUB_SHA') and proof['source'] != os.environ['GITHUB_SHA']:
        raise ValueError('development evidence belongs to another source revision')
    if not proof['files']:
        raise ValueError('development evidence has no source fingerprints')
    for path, digest in proof['files'].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest:
            raise ValueError('development evidence source changed: ' + path)
    if proof['runtime'] != environment(root, python):
        raise ValueError('development evidence interpreter, dependencies or CPU environment changed')
    return proof


def main() -> int:
    """Write the receipt immediately after the workflow's full acceptance suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record(args.root.resolve(), args.output.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
