#!/usr/bin/env python3
"""Execute root documentation fences in retained, local-only fixture projects.

Installer examples use the unpublished wheel and preinstalled dependency matrix.
An identical same-run framework suite may supply a validated execution receipt.
Other development commands execute; no remote repository is contacted.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path

from control_docs import blocks
from development_evidence import COMMAND, verify

PAGES = ['README.md', 'docs/Guides/Quickstart.md', 'docs/SFD_and_CLI.md',
         'docs/Migration.md', 'docs/Backends.md', 'docs/Install_Options.md',
         'docs/Citing_Talos.md', 'CONTRIBUTING.md', 'docs/Developer/Configuration.md']


def checked(command, *, cwd=None, env=None, log=None):
    if log is None:
        result = subprocess.run(command, cwd=cwd, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = result.stdout
    else:
        with Path(log).open('w') as stream:
            result = subprocess.run(command, cwd=cwd, env=env, text=True,
                                    stdout=stream, stderr=subprocess.STDOUT)
        output = Path(log).read_text()
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}): {command!r}\n{output[-10000:]}')
    return output


def dispatch(kind, arguments):
    """Shell functions keep the fixture interpreter after venv activation."""
    if kind == 'python' and list(arguments)[:2] == ['-m', 'pip']:
        return dispatch('pip', arguments[2:])
    virtual = Path(os.environ.get('VIRTUAL_ENV') or os.environ['TALOS_DOC_VENV'])
    python = str(virtual / 'bin/python')
    if kind == 'pip':
        args = list(arguments)
        if not args or args[0] != 'install':
            raise ValueError('Unassigned installer command')
        resolved = []
        for value in args[1:]:
            requirement = re.fullmatch(
                r'talos(\[[A-Za-z0-9_,.-]+\])?(?:\s*@\s*git\+https://github\.com/autonomio/talos(?:@[^\s]+)?)?',
                value)
            if requirement:
                resolved.append(os.environ['TALOS_DOC_WHEEL'] + (requirement[1] or ''))
            elif value.startswith('talos'):
                raise ValueError('Versioned or foreign Talos installs need their own fixture: ' + value)
            else:
                resolved.append(value)
        command = [python, '-m', 'pip', 'install', '--no-index', '--no-deps',
                   '--no-build-isolation', *resolved]
    elif kind == 'python':
        args = list(arguments)
        if args[:2] == ['-m', 'venv']:
            args.insert(2, '--system-site-packages')
        if args[:2] == ['-m', 'build']:
            args.append('--no-isolation')
        command = [python, *args]
    elif kind == 'talos':
        command = [python, '-m', 'talos', *arguments]
    else:
        raise ValueError('Unassigned dispatcher: ' + kind)
    evidence = os.environ.get('TALOS_DEVELOPMENT_EVIDENCE')
    if evidence and kind == 'python' and list(arguments) == COMMAND:
        proof = verify(Path.cwd(), python, Path(evidence))
        print('Reused successful full acceptance suite: ' + str(proof['source']), flush=True)
        reuse = Path(os.environ['TALOS_DEVELOPMENT_REUSE'])
        reuse.write_text(json.dumps({'receipt': evidence, 'proof': proof}, indent=2) + '\n')
        return 0
    capture = kind == 'talos' and arguments and arguments[0] == 'commit'
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.STDOUT)
    if capture:
        sys.stdout.write(result.stdout)
    if kind == 'python' and list(arguments)[:2] == ['-m', 'venv'] and result.returncode == 0:
        created = Path(arguments[-1]).resolve()
        site = next((created / 'lib').glob('python*/site-packages'))
        (site / 'talos-docs-dependencies.pth').write_text(
            '\n'.join(json.loads(os.environ['TALOS_DOC_SITES'])) + '\n')
    if kind == 'talos' and arguments and arguments[0] == 'commit' and result.returncode == 0:
        identifiers = re.findall(r'sha256:[0-9a-f]{64}', result.stdout)
        if not identifiers:
            raise RuntimeError('Successful commit did not return its manifest identity')
        Path(os.environ['TALOS_DOC_STATE']).write_text(identifiers[-1])
    return result.returncode


def shell_prelude():
    command = shlex.quote(sys.executable) + ' ' + shlex.quote(str(Path(__file__).resolve()))
    return '\n'.join([
        'set -e',
        'pip() { ' + command + ' --dispatch pip -- "$@"; }',
        'python() { ' + command + ' --dispatch python -- "$@"; }',
        'talos() { ' + command + ' --dispatch talos -- "$@"; }',
        'coverage() { ' + command + ' --dispatch python -- -m coverage "$@"; }',
        'ruff() { ' + command + ' --dispatch python -- -m ruff "$@"; }',
    ]) + '\n'


def copy_checkout(root, destination):
    shutil.copytree(root, destination, ignore=shutil.ignore_patterns(
        '.git', '__pycache__', '.pytest_cache', '.ruff_cache', '.venv',
        '.coverage', 'dist', 'build', 'results', 'verification-output',
        'node_modules', '.generated', '.docusaurus', 'test-results'))


def migration_fixture(folder):
    """Use the importable paired example independently of README fence order."""
    from examples.keras_to_talos import prepare_data

    fixture = 'from examples.keras_to_talos import existing_model\n'
    callback = folder / 'my_models.py'
    callback.write_text(fixture)
    sys.modules.pop('my_models', None)
    sys.modules.pop('my_sfd', None)
    source = folder / 'examples/keras_to_talos.py'
    return prepare_data(), {
        'path': str(callback), 'source_sha256': hashlib.sha256(fixture.encode()).hexdigest(),
        'source': 'examples/keras_to_talos.py',
        'example_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'context': 'canonical paired Iris example; unchanged five-argument existing_model callback',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, default=Path('/tmp/talos-docs-root.json'))
    args = parser.parse_args()
    root = args.root.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='root-docs-', dir=args.output.parent))
    checkout = work / 'checkout'
    copy_checkout(root, checkout)
    wheels = work / 'wheels'
    wheels.mkdir()
    env = dict(os.environ)
    env.update(KERAS_BACKEND='tensorflow', CUDA_VISIBLE_DEVICES='-1',
               TF_NUM_INTEROP_THREADS='1', TF_NUM_INTRAOP_THREADS='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MPLBACKEND='Agg',
               GIT_AUTHOR_NAME='Talos documentation fixture',
               GIT_AUTHOR_EMAIL='docs-fixture@example.invalid',
               GIT_COMMITTER_NAME='Talos documentation fixture',
               GIT_COMMITTER_EMAIL='docs-fixture@example.invalid',
               PIP_DISABLE_PIP_VERSION_CHECK='1', PYTHONUNBUFFERED='1')
    env.pop('VIRTUAL_ENV', None)
    env['TALOS_DEVELOPMENT_REUSE'] = str(work / 'development-reuse.json')
    env['TALOS_DOC_SOURCE_ROOT'] = str(root)
    build_log = work / 'wheel-build.log'
    checked([sys.executable, '-m', 'build', '--wheel', '--no-isolation',
             '--outdir', str(wheels)], cwd=checkout, env=env, log=build_log)
    wheel = next(wheels.glob('talos-*.whl'))
    virtual = work / 'installer-venv'
    checked([sys.executable, '-m', 'venv', '--without-pip', '--system-site-packages', str(virtual)],
            env=env, log=work / 'installer-venv.log')
    fixture_sites = [value for value in sys.path if value.endswith('site-packages')]
    site_dir = next((virtual / 'lib').glob('python*/site-packages'))
    (site_dir / 'talos-docs-dependencies.pth').write_text('\n'.join(fixture_sites) + '\n')
    checked([str(virtual / 'bin/python'), '-m', 'pip', 'install', 'editables>=0.5,<1'],
            cwd=work, env=env, log=work / 'editable-build-dependency.log')
    checked([str(virtual / 'bin/python'), '-m', 'pip', 'install', '--no-index',
             '--no-deps', str(wheel)], cwd=work, env=env, log=work / 'wheel-install.log')
    env.update(TALOS_DOC_VENV=str(virtual), TALOS_DOC_WHEEL=str(wheel),
               TALOS_DOC_STATE=str(work / 'manifest-id.txt'),
               TALOS_DOC_SITES=json.dumps(fixture_sites))
    os.environ.update({key: value for key, value in env.items()
                       if key.startswith(('TF_', 'KERAS_', 'CUDA_', 'OMP_', 'OPENBLAS_', 'MPL'))})
    sys.path.insert(0, str(root))
    import numpy as np
    import tensorflow as tf

    import talos
    tf.config.set_visible_devices([], 'GPU')
    report = {'schema_version': 1, 'runner': str(Path(__file__).resolve()),
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'root': str(root), 'work_dir': str(work), 'pages': [], 'blocks': [],
              'fixtures': {'installer': 'unpublished local wheel, --no-index --no-deps; preinstalled acceptance dependencies',
                           'wheel': str(wheel), 'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(),
                           'installer_venv': str(virtual), 'build_log': str(build_log),
                           'editable_build_dependency_log': str(work / 'editable-build-dependency.log'),
                           'development': 'complete temporary checkout; commands execute or retain an identical same-run suite receipt',
                           'training': 'real bundled Iris and breast cancer; TensorFlow CPU',
                           'backup': 'temporary local bare Git only; no real remote',
                           'versions': {'talos': talos.__version__, 'numpy': np.__version__,
                                        'tensorflow': tf.__version__}}}
    failures = 0
    original_cwd = Path.cwd()

    def save():
        args.output.write_text(json.dumps(report, indent=2))

    save()
    try:
        for page in PAGES:
            page_blocks, _ = blocks(root / page)
            report['pages'].append({'path': page, 'block_count': len(page_blocks)})
            folder = work / page.replace('/', '_').replace('.md', '')
            folder.mkdir()
            shutil.copytree(root / 'examples', folder / 'examples',
                            ignore=shutil.ignore_patterns('__pycache__'))
            module = types.ModuleType('_talos_docs_' + page.replace('/', '_').replace('.', '_'))
            module.__file__ = str(folder / 'page.py')
            sys.modules[module.__name__] = module
            context = module.__dict__
            context['__name__'] = module.__name__
            sys.path.insert(0, str(folder))
            for imported in list(sys.modules):
                if imported == 'examples' or imported.startswith('examples.'):
                    del sys.modules[imported]
            examples_package = types.ModuleType('examples')
            examples_package.__path__ = [str(folder / 'examples')]
            sys.modules['examples'] = examples_package
            os.chdir(folder)
            if page == 'docs/Migration.md':
                context['my_splits'], report['fixtures']['migration_callback'] = migration_fixture(folder)
            for index, block in enumerate(page_blocks):
                record = {key: value for key, value in block.items() if key != 'source'}
                record.update(path=page, block_index=index, context='sequential page context; documented file saves')
                source_file = folder / f'block-{index:03}.{block["language"]}'
                source_file.write_text(block['source'])
                log = folder / f'block-{index:03}.log'
                record.update(source_file=str(source_file), log=str(log))
                started = time.monotonic()
                try:
                    language = block['language']
                    source = block['source']
                    if language == 'python':
                        destination = None
                        if '# Save as my_sfd.py.' in source or page == 'docs/Migration.md':
                            destination = Path.cwd() / 'my_sfd.py'
                        elif '# Save as manifests/first_sfd.py' in source:
                            destination = Path.cwd() / 'manifests/first_sfd.py'
                        if destination:
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            destination.write_text(source)
                            record['saved_module'] = str(destination)
                        with log.open('w') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                            with tf.device('/CPU:0'):
                                exec(compile(source, str(source_file), 'exec'), context)
                                if 'predictions = result.predict' in source:
                                    assert context['predictions'].shape == (30, 3)
                                    assert np.isfinite(context['result'].data.val_loss).all()
                                if page == 'docs/Guides/Quickstart.md' and 'scan = talos.Scan' in source:
                                    assert len(context['scan'].data) == 2
                                    assert np.isfinite(context['scan'].data.val_loss).all()
                                if page == 'docs/Migration.md':
                                    migrated = talos.run(str(destination), data=context['my_splits'], seed=42,
                                        experiment_name='migration', objective={'metric': 'val_loss', 'direction': 'min'})
                                    assert len(migrated.data) == 2
                                    assert np.isfinite(migrated.data.val_loss).all()
                                    record['context'] = 'canonical paired Iris callback; execute wrapper in two real 5/10-epoch trials'
                                    record['migration_source_sha256'] = report['fixtures']['migration_callback']['example_sha256']
                    elif language in ('sh', 'bash', 'shell'):
                        cwd = Path.cwd()
                        if 'pip install -e' in source or page == 'docs/Developer/Configuration.md':
                            cwd = checkout
                            record['context'] = 'actual full development commands in complete temporary checkout'
                        if 'talos commit ' in source:
                            from ruamel.yaml import YAML
                            yaml = YAML()
                            manifest = cwd / 'manifests/first.yaml'
                            configuration = yaml.load(manifest)
                            configuration['metadata']['mode'] = 'production'
                            with manifest.open('w') as stream:
                                yaml.dump(configuration, stream)
                            bare = cwd.parent / 'study-backup.git'
                            branch = checked(['git', 'symbolic-ref', '--short', 'HEAD'], cwd=cwd, env=env).strip()
                            checked(['git', 'init', '--bare', '--initial-branch', branch, str(bare)],
                                    cwd=cwd, env=env, log=folder / 'local-remote.log')
                            checked(['git', 'remote', 'add', 'origin', str(bare)], cwd=cwd, env=env)
                            config = cwd / 'talos.toml'
                            config.write_text(re.sub(r'backup_remote\s*=\s*[^\n]*',
                                'backup_remote = "../study-backup.git"', config.read_text()))
                            source = source.replace('MANIFEST_ID=sha256:<manifest-hash>',
                                'MANIFEST_ID=$(cat "$TALOS_DOC_STATE")')
                            record['context'] = 'documented production mode/manual hash replacement; local bare remote and store.backup_remote'
                        script = shell_prelude() + source
                        executable = folder / f'block-{index:03}.executed.sh'
                        executable.write_text(script)
                        record['executed_source_file'] = str(executable)
                        checked(['/bin/bash', str(executable)], cwd=cwd, env=env, log=log)
                        reuse = Path(env['TALOS_DEVELOPMENT_REUSE'])
                        if reuse.exists():
                            record['reused_execution'] = json.loads(reuse.read_text())
                            reuse.unlink()
                        if 'talos new my-study' in source:
                            os.chdir(cwd / 'my-study')
                            sys.path.insert(0, str(Path.cwd()))
                        if 'talos backup' in source:
                            assert checked(['git', '--git-dir', str(bare), 'rev-parse', 'HEAD'], env=env).strip() == checked(
                                ['git', 'rev-parse', 'HEAD'], cwd=cwd, env=env).strip()
                            committed = list((cwd / 'manifests/committed').glob('*.yaml'))
                            assert committed
                            record['backup_verified_head'] = checked(['git', 'rev-parse', 'HEAD'], cwd=cwd, env=env).strip()
                        if 'talos run --resume' in source:
                            run_name = 'iris' if page == 'README.md' else 'first'
                            run_dir = max((cwd / 'results/dev').glob(run_name + '_*'), key=lambda path: path.stat().st_mtime)
                            recovered = talos.RunResult.load(run_dir)
                            assert len(recovered.data) == 4
                            assert recovered.data._trial_id.is_unique
                            record['resumed_trials'] = len(recovered.data)
                    elif language == 'yaml':
                        from ruamel.yaml import YAML
                        yaml = YAML()
                        configuration = yaml.load(source)
                        destination = (Path.cwd() / 'experiment.yaml' if page == 'README.md'
                                       else Path.cwd() / 'manifests/first.yaml')
                        destination.write_text(source)
                        from talos.yaml.validator import validate
                        with log.open('w') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                            result = validate(configuration)
                            if hasattr(result, 'valid'):
                                assert result.valid, result
                            record['saved_manifest'] = str(destination)
                            print(result)
                    elif language == 'json':
                        from talos.experiment.feedback_controller import FeedbackController
                        from talos.experiment.msq import MSQ
                        from talos.experiment.param_domain import ParamDomain
                        from talos.experiment.param_search import GridStrategy
                        document = json.loads(source)
                        assert document and document[0]['op'] == 'keep_between'
                        run_dir = max(Path('results/dev').glob('first_*'), key=lambda path: path.stat().st_mtime)
                        destination = run_dir / 'interventions.json'
                        destination.write_text(source)
                        domain = ParamDomain({'learning_rate': [.0001, .001, .01, .1]})
                        strategy = GridStrategy(domain)
                        queue = MSQ(strategy, domain)
                        controller = FeedbackController(feedback_interval=1, intervention_path=destination,
                            audit_log_path=folder / 'intervention-audit.jsonl')
                        with log.open('w') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                            applied = controller.trigger(None, queue, strategy, 1)
                            assert len(applied) == 1
                            assert domain.values_for('learning_rate') == [.001, .01]
                            assert len(list(queue)) == 2
                            print({'applied': applied, 'remaining': domain.params})
                        record['context'] = 'actual intervention file poll and MSQ dispatch; reject outside-range pending trials'
                        record['intervention_file'] = str(destination)
                    else:
                        raise ValueError('Unassigned fence language: ' + language)
                    record['status'] = 'passed'
                except Exception:
                    record['status'] = 'failed'
                    record['error'] = traceback.format_exc()
                    with log.open('a') as stream:
                        stream.write(record['error'])
                    failures += 1
                record['seconds'] = round(time.monotonic() - started, 3)
                report['blocks'].append(record)
                save()
                print(f'{record["status"]}: {page}:{block["start_line"]}', flush=True)
            sys.path.remove(str(folder))
        report['summary'] = {'fences': len(report['blocks']),
            'passed': sum(item['status'] == 'passed' for item in report['blocks']), 'failures': failures}
        save()
    finally:
        os.chdir(original_cwd)
    print(json.dumps(report.get('summary', {})), flush=True)
    return int(failures > 0)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--dispatch':
        kind = sys.argv[2]
        arguments = sys.argv[3:]
        if arguments and arguments[0] == '--':
            arguments = arguments[1:]
        raise SystemExit(dispatch(kind, arguments))
    raise SystemExit(main())
