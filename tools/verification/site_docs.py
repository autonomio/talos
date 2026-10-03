#!/usr/bin/env python3
"""Execute finite documentation-system command fences without training frameworks."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from control_docs import blocks

PAGES = ['docs/Developer/Documentation-System.md', 'docs-site/README.md']


def source_digest(root):
    """Hash every tracked input; generated site output cannot invalidate a receipt."""
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    digest = hashlib.sha256()
    for path in sorted(filter(None, paths)):
        digest.update(path.encode())
        digest.update((root / path).read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    logs = output.parent / 'site-work'
    logs.mkdir(exist_ok=True)
    report = {'schema_version': 1, 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'runtime': {'node': subprocess.check_output(['node', '--version'], text=True).strip(),
                          'npm': subprocess.check_output(['npm', '--version'], text=True).strip()},
              'blocks': [], 'pages': [], 'fixtures': {'execution': 'Real npm installation, security audit, site build and browser checks in the checkout.'}}
    failures = 0
    executions = {}
    for page in PAGES:
        page_blocks, _ = blocks(root / page)
        report['pages'].append({'path': page, 'block_count': len(page_blocks)})
        for index, block in enumerate(page_blocks):
            record = {key: value for key, value in block.items() if key != 'source'}
            record.update(path=page, block_index=index)
            if block['language'] not in ('bash', 'sh', 'shell'):
                raise ValueError(f'Unassigned site fence: {page}:{block["start_line"]}')
            log = logs / f'{Path(page).stem}-{index}.log'
            started = time.monotonic()
            inputs = source_digest(root)
            key = (block['code_sha256'], inputs)
            if key in executions:
                proof = executions[key]
                record.update(status='passed', returncode=0, execution='reused',
                              reused_log=proof['log'], source_sha256=inputs, seconds=0)
            else:
                with log.open('w') as stream:
                    result = subprocess.run(['/bin/bash', '-e', '-c', block['source']],
                                            cwd=root, env=os.environ.copy(), stdout=stream,
                                            stderr=subprocess.STDOUT, timeout=1800)
                record.update(status='passed' if result.returncode == 0 else 'failed',
                              returncode=result.returncode, log=str(log), execution='executed',
                              source_sha256=inputs, seconds=round(time.monotonic() - started, 3))
                failures += int(result.returncode != 0)
                if result.returncode == 0 and source_digest(root) == inputs:
                    executions[key] = record.copy()
            report['blocks'].append(record)
            output.write_text(json.dumps(report, indent=2) + '\n')
            print(f'{record["status"]}: {page}:{block["start_line"]}', flush=True)
    report['summary'] = {'fences': len(report['blocks']), 'failures': failures}
    output.write_text(json.dumps(report, indent=2) + '\n')
    return int(failures != 0)


if __name__ == '__main__':
    raise SystemExit(main())
