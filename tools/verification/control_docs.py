#!/usr/bin/env python3
"""Execute documented control/command pages with real held-out Iris data."""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import traceback

PAGES = ['Analyze', 'Predict', 'Evaluate', 'Deploy', 'Restore', 'Scan', 'Parallelism',
         'Custom_Reducers', 'Probabilistic_Reduction', 'Optimization_Strategies',
         'Local_Strategy', 'Gamify', 'Monitoring', 'Energy_Draw', 'Overview', 'Workflow', 'Roadmap']


def blocks(path):
    lines = path.read_text().splitlines()
    found, fenced = [], set()
    index = 0
    while index < len(lines):
        if lines[index].startswith('```'):
            start, language = index + 1, lines[index][3:].strip()
            end = index + 1
            while end < len(lines) and not lines[end].startswith('```'):
                end += 1
            source = '\n'.join(lines[index + 1:end]) + '\n'
            found.append({'start_line': start, 'language': language, 'source': source,
                          'code_sha256': hashlib.sha256(source.encode()).hexdigest()})
            fenced.update(range(index + 1, end + 2))
            index = end
        index += 1
    inline = []
    for number, line in enumerate(lines, 1):
        if number not in fenced:
            inline.extend({'start_line': number, 'source': match,
                           'code_sha256': hashlib.sha256((match + '\n').encode()).hexdigest()}
                          for match in re.findall(r'`([^`]+)`', line))
    return found, inline


def postconditions(page, context):
    import numpy as np
    if page == 'Analyze':
        assert context['r'].rounds() == 2
        assert np.isfinite(context['r'].high('val_accuracy'))
    elif page == 'Predict':
        assert context['probabilities'].shape == (30, 3)
        np.testing.assert_allclose(context['probabilities'].sum(axis=-1), 1, atol=1e-5)
    elif page == 'Evaluate':
        assert len(context['scores']) == 3
        assert all(0 <= score <= 1 for score in context['scores'])
    elif page in ('Deploy', 'Restore', 'Overview'):
        import talos
        from talos.backends import backend_for
        archive = (context['deployment'].path if page != 'Overview' else 'deployed_package.zip')
        restored = context.get('restore') if page == 'Restore' else talos.Restore(archive)
        selected = context['scan_object'].best_model('val_loss', asc=True)
        expected = backend_for(selected).predict(selected, context['x_test'])
        actual = backend_for(restored.model).predict(restored.model, context['x_test'])
        np.testing.assert_allclose(expected, actual, atol=1e-6)
        assert len(restored.results) == len(context['scan_object'].data)
    elif page == 'Scan':
        assert len(context['limited'].data) == 3
        assert len(context['scan_object'].data) == 1
        assert len(context['out'].history['loss']) == 1
    elif page == 'Custom_Reducers':
        assert context['custom_scan'].data.first_neuron.tolist() == [4]
    elif page == 'Local_Strategy':
        assert context['local_scan'].reduction_threshold == .3
        assert 'legacy_local_control_change' in (context['local_scan'].run_dir / 'audit.jsonl').read_text()
    elif page == 'Gamify':
        assert len(context['gamify_scan'].data) == 1
        assert len(context['gamify_resumed'].data) == 1, 'Paused JSON edit must remove the pending disabled trial before training'
    elif page == 'Monitoring':
        logs = list(Path('epoch_logs').glob('epochs-*.log'))
        assert logs and len(logs[0].read_text().splitlines()) == 3
        assert len(context['history'].history['loss']) == 2
    elif page == 'Energy_Draw':
        history = context['history'].history
        assert history['watts_min'] == history['watts_max'] == [10.0]
        assert history['Ws'][0] == round(10 * history['seconds'][0], 2)
    elif page == 'Parallelism':
        worker_scans = context['worker_scans']
        assert sum(len(scan.data) for scan in worker_scans) == 2
        assert len({trial for scan in worker_scans for trial in scan.data._trial_id}) == 2
    elif page == 'Optimization_Strategies':
        assert len(context['random_scan'].data) == 1
        assert 1 <= len(context['out'].history['loss']) <= 2
    elif page == 'Probabilistic_Reduction':
        assert 1 <= len(context['reduced'].data) <= 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, default=Path('/tmp/talos-docs-controls-search.json'))
    parser.add_argument('--pages', nargs='*', default=PAGES)
    parser.add_argument('--backend', choices=['torch', 'tensorflow'], default='torch')
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    os.environ['KERAS_BACKEND'] = args.backend
    os.environ['KERAS_TORCH_DEVICE'] = 'cpu'
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
    import keras
    import numpy as np
    import talos
    if args.backend == 'tensorflow':
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
    device_scope = keras.device if hasattr(keras, 'device') else lambda device: contextlib.nullcontext()
    work = Path(tempfile.mkdtemp(prefix='talos-doc-controls-'))
    fixture_source = blocks(root / 'docs/Scan.md')[0][0]['source']
    fixture_file = work / 'iris_setup.py'
    fixture_file.write_text(fixture_source)
    fixture_hash = hashlib.sha256(fixture_source.encode()).hexdigest()
    provider_dir = work / 'provider-bin'
    provider_dir.mkdir()
    provider = provider_dir / 'nvidia-smi'
    provider.write_text('#!/bin/sh\ncase "$*" in\n *power.draw*) echo 10.0 ;;\n *) echo "NVIDIA provider fixture: CPU verification; no physical GPU measurement" ;;\nesac\n')
    provider.chmod(0o755)
    report = {'schema_version': 1, 'runner': str(Path(__file__).resolve()),
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'root': str(root), 'work_dir': str(work), 'pages': [], 'blocks': [], 'inline_commands': [],
              'fixtures': {'iris_setup': {'path': str(fixture_file), 'source_sha256': fixture_hash,
                            'source': 'docs/Scan.md:' + str(blocks(root / 'docs/Scan.md')[0][0]['start_line']), 'real_data': 'sklearn bundled Iris',
                            'split': '90 train / 30 validation / 30 held-out test, train-only scaler'},
                           'hardware_provider': {'path': str(provider), 'physical_measurement': False},
                           'keras_backend': args.backend, 'device': 'CPU'},
              'versions': {'talos': talos.__version__, 'keras': keras.__version__, 'numpy': np.__version__}}
    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    old_cwd = Path.cwd()
    failures = 0
    try:
        for page in args.pages:
            page_path = root / 'docs' / (page + '.md')
            page_blocks, inline = blocks(page_path)
            folder = work / page
            folder.mkdir()
            os.chdir(folder)
            context = {'__name__': '__main__'}
            fixture_log = folder / 'fixture.log'
            page_result = {'path': str(page_path.relative_to(root)), 'block_count': len(page_blocks),
                           'fixture': str(fixture_file), 'fixture_log': str(fixture_log)}
            report['pages'].append(page_result)
            try:
                if page != 'Scan' and page_blocks:
                    with fixture_log.open('w') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                        with device_scope('cpu'):
                            exec(compile(fixture_source, str(fixture_file), 'exec'), context)
            except Exception:
                page_result['fixture_status'] = 'failed'
                page_result['error'] = traceback.format_exc()
                failures += 1
                save()
                continue
            for index, block in enumerate(page_blocks):
                record = {key: value for key, value in block.items() if key != 'source'}
                record.update(path=str(page_path.relative_to(root)), block_index=index,
                              context='page order; shared held-out Iris setup' if page != 'Scan' else 'page order from complete minimal setup')
                source_file = folder / f'block-{index:03}.py'
                source_file.write_text(block['source'])
                log_file = folder / f'block-{index:03}.log'
                record['source_file'] = str(source_file)
                record['log'] = str(log_file)
                started = time.monotonic()
                try:
                    if block['language'] != 'python':
                        raise ValueError('Unassigned non-Python code fence: ' + block['language'])
                    with log_file.open('w') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                        with device_scope('cpu'):
                            exec(compile(block['source'], str(source_file), 'exec'), context)
                            if page == 'Scan' and block['source'].startswith('def input_model'):
                                values = {key: candidate[0] for key, candidate in context['p'].items()}
                                history, model = context['input_model'](context['x'], context['y'], context['x_val'], context['y_val'], values)
                                assert len(history.history['loss']) == values['epochs']
                    record['status'] = 'passed'
                except Exception:
                    record['status'] = 'failed'
                    record['error'] = traceback.format_exc()
                    with log_file.open('a') as stream:
                        stream.write(record['error'])
                    failures += 1
                record['seconds'] = round(time.monotonic() - started, 3)
                report['blocks'].append(record)
                save()
            try:
                with device_scope('cpu'):
                    postconditions(page, context)
                page_result['postconditions'] = 'passed'
            except Exception:
                page_result['postconditions'] = 'failed'
                page_result['error'] = traceback.format_exc()
                failures += 1
            for item in inline:
                source = item['source']
                material = (source == 'PowerDraw(device=0)' or
                            bool(re.match(r'^(?:pip|pip3|python|python3|export|watch|nvidia-smi|git|talos)(?:\s|$)', source))
                            and source != 'talos')
                if not material:
                    continue
                record = {**item, 'path': str(page_path.relative_to(root)), 'kind': 'inline_command'}
                try:
                    if source == 'nvidia-smi':
                        output = subprocess.run([str(provider)], check=True, capture_output=True, text=True).stdout
                        record.update(status='passed', context='explicit local NVIDIA command provider; no physical GPU validation', result=output.strip())
                    elif source == 'PowerDraw(device=0)':
                        previous_path = os.environ.get('PATH', '')
                        os.environ['PATH'] = str(provider_dir) + os.pathsep + previous_path
                        try:
                            power = eval(source, {**context, 'PowerDraw': talos.callbacks.PowerDraw})
                            power.on_train_begin()
                            power.on_epoch_begin(0)
                            power.on_epoch_end(0)
                            assert power.log['epoch_begin'] == power.log['epoch_end'] == [10.0]
                        finally:
                            os.environ['PATH'] = previous_path
                        record.update(status='passed', context='real PowerDraw callback, explicit nvidia-smi provider fixture; no physical energy claim')
                    else:
                        raise ValueError('Material inline command needs an explicit execution fixture: ' + source)
                except Exception:
                    record.update(status='failed', error=traceback.format_exc())
                    failures += 1
                report['inline_commands'].append(record)
            page_result['status'] = 'passed' if page_result.get('postconditions') == 'passed' and all(
                item['status'] == 'passed' for item in report['blocks'] if item['path'] == page_result['path']) else 'failed'
            print(page, page_result['status'], len(page_blocks), flush=True)
            save()
            from matplotlib import pyplot as plt
            plt.close('all')
        report['summary'] = {'blocks': len(report['blocks']), 'passed': sum(item['status'] == 'passed' for item in report['blocks']),
                             'inline_commands': len(report['inline_commands']), 'failures': failures}
        save()
    finally:
        os.chdir(old_cwd)
    print(json.dumps(report['summary']), flush=True)
    return int(bool(failures))

if __name__ == '__main__':
    raise SystemExit(main())
