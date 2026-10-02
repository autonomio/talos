#!/usr/bin/env python3
"""Execute all documentation fences and examples, then reject missing or stale evidence."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def inventory(root):
    fences, examples = [], []
    docs_map = root / 'docs-site/docs-map.json'
    if docs_map.exists():
        pages = [root / entry['source'] for entry in json.loads(docs_map.read_text())['documents']]
    else:
        pages = [root / 'README.md', root / 'CONTRIBUTING.md', *sorted((root / 'docs').rglob('*.md'))]
    for path in pages:
        lines = path.read_text().splitlines()
        index = 0
        while index < len(lines):
            if lines[index].startswith('```'):
                language, start = lines[index][3:].strip(), index + 1
                end = index + 1
                while end < len(lines) and not lines[end].startswith('```'):
                    end += 1
                if end == len(lines):
                    raise ValueError(f'Unclosed fence: {path}:{start}')
                source = '\n'.join(lines[index + 1:end]) + '\n'
                fences.append({'path': str(path.relative_to(root)), 'start_line': start,
                               'language': language, 'code_sha256': hashlib.sha256(source.encode()).hexdigest()})
                index = end
            index += 1
    for path in sorted((root / 'examples').rglob('*')):
        if path.suffix not in ('.py', '.ipynb'):
            continue
        record = {'path': str(path.relative_to(root)), 'whole_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if path.suffix == '.ipynb':
            notebook = json.loads(path.read_text())
            record['cells'] = [{'index': index, 'code_sha256': hashlib.sha256(''.join(cell['source']).encode()).hexdigest()}
                               for index, cell in enumerate(notebook['cells']) if cell['cell_type'] == 'code']
        examples.append(record)
    return {'fences': fences, 'examples': examples}


def verify(root, output, selected, execution):
    from tools.verification.development_evidence import source_manifest
    expected = inventory(root)
    actual, example_records, providers = {}, {}, []
    errors = []
    for name in ('api', 'control', 'root', 'examples', 'site'):
        receipt = output / f'{name}.json'
        if not receipt.exists():
            continue
        report = json.loads(receipt.read_text())
        if report.get('child_failures') or report.get('failed') or report.get('summary', {}).get('failures'):
            errors.append(f'Failed component receipt: {name}')
        if name == 'api' and any(block['path'] == 'docs/Examples_PyTorch_Code.md' for block in report.get('blocks', [])):
            archive = report.get('portable_archive_check') or {}
            expected_archive = next(fence for fence in expected['fences'] if fence['path'] == 'docs/Examples_PyTorch_Code.md')
            if archive.get('status') != 'passed' or archive.get('source_code_sha256') != expected_archive['code_sha256']:
                errors.append('Missing, failed or stale guarded Torch archive check')
        for block in report.get('blocks', []):
            reuse = block.get('reused_execution')
            if reuse:
                proof = reuse['proof']
                if source_manifest(root) != proof['files']:
                    errors.append('Stale reused development source inventory')
                for path, digest in proof['files'].items():
                    if hashlib.sha256((root / path).read_bytes()).hexdigest() != digest:
                        errors.append('Stale reused development source: ' + path)
            key = (block['path'], block['start_line'])
            if key in actual:
                errors.append(f'Duplicate receipt: {key}')
            actual[key] = block
        for record in report.get('examples', []):
            if record['path'] in example_records:
                errors.append(f'Duplicate example receipt: {record["path"]}')
            example_records[record['path']] = record
        providers.extend(report.get('inline_commands', []))
    missing = []
    for fence in expected['fences']:
        record = actual.pop((fence['path'], fence['start_line']), None)
        if record is None:
            missing.append(fence)
        elif record.get('code_sha256') != fence['code_sha256'] or record.get('status') != 'passed':
            errors.append(f'Failed or stale fence: {fence["path"]}:{fence["start_line"]}')
    if actual:
        errors.append(f'Stale/unassigned fence receipts: {list(actual)}')
    for example in expected['examples']:
        record = example_records.pop(example['path'], None)
        if record is None:
            missing.append(example)
            continue
        if record.get('whole_file_sha256') != example['whole_file_sha256'] or record.get('status') != 'passed':
            errors.append(f'Failed or stale example: {example["path"]}')
        if 'cells' in example:
            detail = {cell['index']: cell for cell in record.get('evidence', {}).get('cells', []) if cell['cell_type'] == 'code'}
            if set(detail) != {cell['index'] for cell in example['cells']}:
                errors.append(f'Missing or extra notebook cells: {example["path"]}')
            for cell in example['cells']:
                evidence = detail.get(cell['index'], {})
                if evidence.get('code_sha256') != cell['code_sha256'] or evidence.get('status') != 'passed':
                    errors.append(f'Failed or stale notebook cell: {example["path"]}:{cell["index"]}')
    if example_records:
        errors.append(f'Stale/unassigned example receipts: {list(example_records)}')
    errors.extend(f'Failed inline command: {record["path"]}:{record["start_line"]}' for record in providers if record['status'] != 'passed')
    errors.extend(f'Executor failed: {name}' for name, record in execution.items() if record['returncode'])
    versions = {}
    for name in ('talos', 'tensorflow', 'keras', 'torch', 'numpy', 'protobuf', 'scikit-learn'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    import talos
    result = {'schema_version': 1, 'talos_source_version': talos.__version__, 'root': str(root), 'python': sys.version.split()[0], 'versions': versions,
              'selected_components': selected, 'inventory': expected, 'executors': execution,
              'complete': not missing and not errors, 'missing': missing, 'errors': errors,
              'counts': {'fences': len(expected['fences']), 'examples': len(expected['examples']),
                         'notebook_code_cells': sum(len(e.get('cells', [])) for e in expected['examples']),
                         'inline_commands': len(providers)},
              'hardware_scope': 'CPU training. NVIDIA power/command paths use an explicitly declared provider; physical hardware and remote entropy services are not verified.'}
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('complete', 'counts', 'errors')}, indent=2), flush=True)
    return int(bool(errors or (missing and len(selected) == 5)))


def execute_component(command, root, env, log, deadline):
    """Use the shared deadline and kill the entire executor process group on expiry."""
    start = time.monotonic()
    remaining = deadline - start
    if remaining <= 0:
        raise TimeoutError('Documentation total deadline expired before component launch')
    with log.open('w') as stream:
        with subprocess.Popen(command, env=env, cwd=root, stdout=stream,
                              stderr=subprocess.STDOUT, start_new_session=True) as process:
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise TimeoutError('Documentation total deadline expired during component execution')
    return {'returncode': process.returncode, 'seconds': round(time.monotonic() - start, 3), 'log': str(log)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path, default=Path('verification-output'))
    parser.add_argument('--components', nargs='+', choices=['api', 'control', 'root', 'examples', 'site'],
                        default=['api', 'control', 'root', 'examples', 'site'])
    parser.add_argument('--timeout-seconds', type=int, default=1800,
                        help='total deadline for all queued and running component processes')
    parser.add_argument('--development-evidence', type=Path,
                        help='same-run successful full framework suite receipt')
    parser.add_argument('--verify-only', action='store_true', help='Verify current hashes against previously generated receipts.')
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error('--timeout-seconds must be positive')
    root, output = args.root.resolve(), args.output_dir.resolve()
    sys.path.insert(0, str(root))
    output.mkdir(parents=True, exist_ok=True)
    helpers = root / 'tools/verification'
    env = os.environ.copy()
    env.update({'PYTHONPATH': os.pathsep.join(filter(None, (str(root), env.get('PYTHONPATH')))),
                'KERAS_BACKEND': 'tensorflow', 'KERAS_TORCH_DEVICE': 'cpu', 'CUDA_VISIBLE_DEVICES': '-1',
                'TF_NUM_INTRAOP_THREADS': '1', 'TF_NUM_INTEROP_THREADS': '1', 'OMP_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1', 'MPLBACKEND': 'Agg', 'TALOS_DOCS_ROOT': str(root),
                'TALOS_DOCS_WORK': str(output / 'api-work'), 'TALOS_DOCS_MANIFEST': str(output / 'api.json')})
    commands = {
        'api': [sys.executable, str(helpers / 'api_docs.py')],
        'control': [sys.executable, str(helpers / 'control_docs.py'), '--root', str(root), '--output', str(output / 'control.json')],
        'root': [sys.executable, str(helpers / 'root_docs.py'), '--root', str(root), '--output', str(output / 'root.json')],
        'examples': [sys.executable, str(helpers / 'examples.py'), '--root', str(root), '--output', str(output / 'examples.json')],
        'site': [sys.executable, str(helpers / 'site_docs.py'), '--root', str(root), '--output', str(output / 'site.json')],
    }
    if args.development_evidence:
        env['TALOS_DEVELOPMENT_EVIDENCE'] = str(args.development_evidence.resolve())
    deadline = time.monotonic() + args.timeout_seconds
    prior = output / 'manifest.json'
    execution = json.loads(prior.read_text()).get('executors', {}) if prior.exists() else {}

    def execute(name):
        receipt = output / f'{name}.json'
        if receipt.exists():
            receipt.unlink()
        return execute_component(commands[name], root, env, output / f'{name}.log', deadline)
    if not args.verify_only:
        print('Running ' + ', '.join(args.components) + '; logs: ' + str(output), flush=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = {pool.submit(execute, name): name for name in args.components}
            for job in as_completed(jobs):
                name = jobs[job]
                try:
                    execution[name] = job.result()
                except Exception as exc:
                    execution[name] = {'returncode': 1, 'error': str(exc)}
                print(name, execution[name], flush=True)
    return verify(root, output, args.components, execution)


if __name__ == '__main__':
    raise SystemExit(main())
