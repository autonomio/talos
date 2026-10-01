#!/usr/bin/env python3
"""Execute finite documentation-system command fences without training frameworks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from control_docs import blocks

PAGES = ['docs/Developer/Documentation-System.md', 'docs-site/README.md']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    logs = output.parent/'site-work'
    logs.mkdir(exist_ok=True)
    report = {'schema_version': 1, 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'runtime': {'node': subprocess.check_output(['node', '--version'], text=True).strip(),
                          'npm': subprocess.check_output(['npm', '--version'], text=True).strip()},
              'blocks': [], 'pages': [], 'fixtures': {'execution': 'Real npm installation, security audit, site build and browser checks in the checkout.'}}
    failures = 0
    for page in PAGES:
        page_blocks, _ = blocks(root/page)
        report['pages'].append({'path': page, 'block_count': len(page_blocks)})
        for index, block in enumerate(page_blocks):
            record = {key: value for key, value in block.items() if key != 'source'}
            record.update(path=page, block_index=index)
            if block['language'] not in ('bash', 'sh', 'shell'):
                raise ValueError(f'Unassigned site fence: {page}:{block["start_line"]}')
            log = logs/f'{Path(page).stem}-{index}.log'
            started = time.monotonic()
            with log.open('w') as stream:
                result = subprocess.run(['/bin/bash', '-e', '-c', block['source']],
                                        cwd=root, env=os.environ.copy(), stdout=stream,
                                        stderr=subprocess.STDOUT, timeout=1800)
            record.update(status='passed' if result.returncode == 0 else 'failed',
                          returncode=result.returncode, log=str(log),
                          seconds=round(time.monotonic()-started, 3))
            failures += int(result.returncode != 0)
            report['blocks'].append(record)
            output.write_text(json.dumps(report, indent=2)+'\n')
            print(f'{record["status"]}: {page}:{block["start_line"]}', flush=True)
    report['summary'] = {'fences': len(report['blocks']), 'failures': failures}
    output.write_text(json.dumps(report, indent=2)+'\n')
    return int(failures != 0)


if __name__ == '__main__':
    raise SystemExit(main())
