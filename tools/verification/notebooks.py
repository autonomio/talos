"""Execute every notebook code cell unchanged, with a separate Python process per notebook."""
import argparse
import hashlib
import json
import os
import tempfile
import time
import traceback
from pathlib import Path


def run_notebook(path, report_path):
    path = Path(path).resolve()
    notebook = json.loads(path.read_text())
    scope = {'__name__': '__talos_notebook__'}
    report = {'file': str(path), 'whole_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'cells': [], 'status': 'running'}
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix='talos-notebook-') as work:
        os.chdir(work)
        try:
            for index, cell in enumerate(notebook['cells']):
                source = ''.join(cell['source'])
                entry = {'index': index, 'cell_type': cell['cell_type'],
                         'code_sha256': hashlib.sha256(source.encode()).hexdigest()}
                report['cells'].append(entry)
                if cell['cell_type'] != 'code':
                    entry['status'] = 'prose'
                    continue
                started = time.monotonic()
                try:
                    exec(compile(source, f'{path}:cell-{index}', 'exec'), scope)
                    entry['status'] = 'passed'
                except BaseException:
                    entry['status'] = 'failed'
                    entry['traceback'] = traceback.format_exc()
                    report['status'] = 'failed'
                    raise
                finally:
                    entry['elapsed_seconds'] = time.monotonic() - started
                    Path(report_path).write_text(json.dumps(report, indent=2) + '\n')
            report['status'] = 'passed'
        finally:
            os.chdir(previous)
            Path(report_path).write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('notebook')
    parser.add_argument('report')
    arguments = parser.parse_args()
    run_notebook(arguments.notebook, arguments.report)
