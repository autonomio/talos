from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Final

import yaml
from _common import loads_toml

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
CONFIG_PATH: Final[Path] = REPO_ROOT / 'governance.yml'
WORKFLOWS_DIR: Final[Path] = REPO_ROOT / '.github/workflows'
RULESET_PATH: Final[Path] = REPO_ROOT / '.github/rulesets/master.json'


def _mapping(value: object, name: str) -> dict[str, object]:
    assert isinstance(value, dict), f'{name} must be a mapping'
    return {str(key): item for key, item in value.items()}


def _config() -> dict[str, object]:
    return _mapping(yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8')), 'governance.yml')


def _section(name: str) -> dict[str, object]:
    return _mapping(_config().get(name), name)


def _str(section: dict[str, object], key: str) -> str:
    value = section.get(key)
    assert isinstance(value, str), f'{key} must be a string'
    return value


def _int(section: dict[str, object], key: str) -> int:
    value = section.get(key)
    assert isinstance(value, int), f'{key} must be an integer'
    return value


def _str_list(section: dict[str, object], key: str) -> list[str]:
    value = section.get(key)
    assert isinstance(value, list), f'{key} must be a list'
    assert all(isinstance(item, str) for item in value), f'{key} must contain strings'
    return [item for item in value if isinstance(item, str)]


def _required_status_contexts() -> list[str]:
    payload = json.loads(RULESET_PATH.read_text(encoding='utf-8'))
    rules = payload['rules']
    assert isinstance(rules, list)
    for rule in rules:
        rule_map = _mapping(rule, 'ruleset rule')
        if rule_map.get('type') != 'required_status_checks':
            continue
        params = _mapping(rule_map.get('parameters'), 'required_status_checks parameters')
        checks = params.get('required_status_checks')
        assert isinstance(checks, list)
        contexts: list[str] = []
        for check in checks:
            check_map = _mapping(check, 'required status check')
            context = check_map.get('context')
            assert isinstance(context, str)
            contexts.append(context)
        return contexts
    raise AssertionError('required_status_checks rule missing from ruleset snapshot')


# The packaging gate builds on the pinned interpreter and then proves the
# wheel installs on every supported one. Every other workflow pins.
MULTI_INTERPRETER_WORKFLOWS: frozenset[str] = frozenset({'pr_checks_packaging.yml', 'ci.yml', 'security.yml'})


def _setup_python_versions() -> dict[str, list[str]]:
    versions: dict[str, list[str]] = {}
    for workflow in sorted(WORKFLOWS_DIR.glob('*.yml')):
        workflow_payload = _mapping(
            yaml.safe_load(workflow.read_text(encoding='utf-8')), workflow.name
        )
        jobs = _mapping(workflow_payload.get('jobs'), f'{workflow.name}.jobs')
        workflow_versions: list[str] = []
        for job_name, job in jobs.items():
            job_map = _mapping(job, f'{workflow.name}.{job_name}')
            steps = job_map.get('steps')
            assert isinstance(steps, list), f'{workflow.name}.{job_name}.steps must be a list'
            for step in steps:
                step_map = _mapping(step, f'{workflow.name}.{job_name}.step')
                uses = step_map.get('uses')
                if not isinstance(uses, str) or not uses.startswith('actions/setup-python@'):
                    continue
                with_config = _mapping(step_map.get('with'), f'{workflow.name}.{job_name}.with')
                version = with_config.get('python-version')
                assert isinstance(version, str), f'{workflow.name} python-version must be quoted'
                workflow_versions.append(version)
        if workflow_versions:
            versions[workflow.name] = workflow_versions
    return versions


def _pinned_dev_tool(pyproject: dict[str, object], package: str) -> str:
    """The version `pyproject.toml` pins for one dev tool.

    This is the only place a tool version is written. `dev-env.in` resolves
    `.[dev]`, so the compiled set derives from here rather than restating it,
    and a bump has one file to find.
    """
    project = _mapping(pyproject.get('project'), 'pyproject [project]')
    extras = _mapping(project.get('optional-dependencies'), '[project.optional-dependencies]')
    dev = extras.get('dev')
    assert isinstance(dev, list), 'the dev extra must be a list'
    pins = [
        entry.split('==', 1)[1]
        for entry in dev
        if isinstance(entry, str) and entry.startswith(f'{package}==')
    ]
    assert len(pins) == 1, f'{package} must be pinned exactly once in the dev extra, got {pins}'
    return pins[0]


def _requirement_pins(package: str) -> list[str]:
    # Only the compiled set carries a literal pin now: `dev-env.in` resolves
    # `.[dev]`, so the version it installs comes from pyproject. Reading both
    # still catches a hand-edited compiled set, which would otherwise ship an
    # ungoverned tool while every source it derives from reads correctly.
    sources = [
        REPO_ROOT / 'requirements' / 'ci' / 'dev-env.in',
        REPO_ROOT / 'requirements' / 'ci' / 'dev-env.txt',
    ]
    pins = {
        pin
        for source in sources
        for pin in re.findall(
            rf'^{package}==([0-9.]+)\b', source.read_text(encoding='utf-8'), re.MULTILINE
        )
    }
    return sorted(pins)


def test_an_exactly_pinned_tool_is_not_also_given_a_range() -> None:
    """A tool pinned exactly lives in `pyproject.toml` and nowhere else.

    Restating an exact pin as a range in `constraints.txt` documents nothing,
    adds a second place a bump has to find, and is easy to get wrong in a way
    nothing catches -- nothing installs from that file. `pyright>=1.1` read
    like a floor while admitting every release pyright has ever published,
    because they are all 1.1.x.
    """
    pyproject = loads_toml((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    project = _mapping(pyproject.get('project'), 'pyproject [project]')
    extras = _mapping(project.get('optional-dependencies'), '[project.optional-dependencies]')
    dev = extras.get('dev')
    assert isinstance(dev, list)
    pinned = sorted(
        entry.split('==', 1)[0]
        for entry in dev
        if isinstance(entry, str) and '==' in entry
    )
    assert pinned, 'no dev tool is pinned exactly; this guard would be vacuous'

    constraints = (REPO_ROOT / 'requirements' / 'constraints.txt').read_text(encoding='utf-8')
    restated = [
        package for package in pinned
        if re.search(rf'^{re.escape(package)}\s*[><=]', constraints, re.MULTILINE)
    ]
    assert not restated, (
        f'{restated} are pinned exactly in pyproject.toml and also given a range in '
        f'constraints.txt. Two declarations of one version is what this slice removed.'
    )


def test_governance_config_schema_is_minimal() -> None:
    config = _config()

    assert config['schema_version'] == 2
    assert set(config) == {
        'schema_version',
        'repository',
        'adoption',
        'layout',
        'runtime',
        'review',
        'automation',
        'slice',
        'commits',
        'changelog',
        'bootstrap',
        'ruleset',
        'gates',
    }


def _gates() -> dict[str, dict[str, object]]:
    return {
        name: _mapping(body, f'gates.{name}')
        for name, body in _section('gates').items()
    }


def _required_contexts_from_config() -> list[str]:
    # A gate owes a required status check only when it both runs and blocks.
    return sorted(
        _str(body, 'context')
        for body in _gates().values()
        if body.get('enabled', True) is not False and body.get('required') is True
    )


def test_ruleset_required_checks_match_config() -> None:
    """The ruleset snapshot lists exactly the gates configured to block.

    The snapshot is what `pr_checks_ruleset` holds live branch protection to,
    so letting it drift from `gates.*.required` would let a gate be marked
    blocking in config while nothing on `main` actually required it.
    """
    ruleset_snapshot = json.loads(RULESET_PATH.read_text(encoding='utf-8'))

    assert ruleset_snapshot['name'] == _str(_section('ruleset'), 'name')
    assert sorted(_required_status_contexts()) == _required_contexts_from_config()


def test_workflow_runtime_and_tooling_match_config() -> None:
    runtime = _section('runtime')
    python_version = _str(runtime, 'python_version')
    pyproject = loads_toml((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
    # pyproject is the only place a tool version is written; the compiled
    # requirement sets are derived from it, so they are checked against it
    # rather than against a second declaration.
    ruff_version = _pinned_dev_tool(pyproject, 'ruff')
    pyright_version = _pinned_dev_tool(pyproject, 'pyright')

    assert _setup_python_versions()
    for workflow_name, versions in _setup_python_versions().items():
        if workflow_name in MULTI_INTERPRETER_WORKFLOWS:
            # The install matrix exists to prove the wheel imports on every
            # supported interpreter, so it cannot pin one. Matrix expressions
            # are not literals; every literal that remains must still be the
            # version governance.yml names.
            literals = [v for v in versions if not v.startswith('${{')]
            assert set(literals) <= {'3.10', '3.11', '3.12', '3.13'}, (workflow_name, literals)
            continue
        assert versions == [python_version] * len(versions), workflow_name
    assert _requirement_pins('ruff') == [ruff_version]
    assert _requirement_pins('pyright') == [pyright_version]
    assert pyproject['project']['requires-python'] == '>=3.10,<3.14'
    assert pyproject['tool']['pyright']['pythonVersion'] == '3.10'


def test_bootstrap_review_and_slice_settings_match_config() -> None:
    bootstrap = _section('bootstrap')
    review = _section('review')
    slice_config = _section('slice')
    variables = _mapping(bootstrap.get('variables'), 'bootstrap.variables')
    secrets = _mapping(bootstrap.get('secrets'), 'bootstrap.secrets')
    bootstrap_workflow = (WORKFLOWS_DIR / 'bootstrap_repository.yml').read_text(encoding='utf-8')
    slice_workflow = (WORKFLOWS_DIR / 'pr_checks_slice.yml').read_text(encoding='utf-8')
    slice_issue_workflow = (WORKFLOWS_DIR / 'pr_checks_slice_on_issue.yml').read_text(
        encoding='utf-8'
    )
    label_workflow = (WORKFLOWS_DIR / 'copy-standard-labels.yml').read_text(encoding='utf-8')
    assert 'configure_repository.py' in label_workflow
    assert (REPO_ROOT / '.github/labels.json').is_file()
    laws = (REPO_ROOT / 'CLAUDE.md').read_text(encoding='utf-8')
    setup = (REPO_ROOT / 'SETUP.md').read_text(encoding='utf-8')
    issue_template = (REPO_ROOT / _str(slice_config, 'issue_template')).read_text(encoding='utf-8')

    assert f'timeout-minutes: {_int(bootstrap, "timeout_minutes")}' in bootstrap_workflow
    assert f'labels:\n  - {_str(slice_config, "label")}' in issue_template
    assert _str(slice_config, 'issue_template') in slice_workflow
    assert _str(slice_config, 'issue_template') in slice_issue_workflow
    assert _str(variables, 'ruleset_id') in setup
    assert _str(secrets, 'bootstrap_token') in bootstrap_workflow
    assert _str(secrets, 'ruleset_audit_token') in setup
    assert _str(review, 'approving_authority') in laws
    assert _str(review, 'approving_authority') in setup


def test_every_gate_section_is_read_by_a_gate() -> None:
    """Every gate section must be one some gate actually reads.

    A typo'd or orphaned section is worse than a missing one: it looks like
    configuration and changes nothing, so a repository believes it has tuned
    a gate that is still running its default.
    """
    governance_source = '\n'.join(
        path.read_text(encoding='utf-8')
        for path in sorted((REPO_ROOT / 'governance').glob('*.py'))
    )
    workflow_source = '\n'.join(
        path.read_text(encoding='utf-8') for path in sorted(WORKFLOWS_DIR.glob('*.yml'))
    )
    control = {'enabled', 'required', 'context'}
    for name, body in sorted(_gates().items()):
        if not set(body) - control:
            # Control-surface-only: consumed generically by the honesty gate
            # and the workflows, so there is no by-name reader to look for.
            continue
        # Matched on the quoted gate name rather than on a specific call
        # form: the readers wrap across lines, and a grep for one spelling
        # would pass by accident the day someone reformats the call.
        read_by = f"'{name}'" in governance_source or f'gates.{name}.' in workflow_source
        assert read_by, f'gates.{name} sets policy but nothing reads it'


def test_every_layout_key_is_read_by_a_gate() -> None:
    """Same rule for `layout`: a path nobody resolves is decoration."""
    governance_source = '\n'.join(
        path.read_text(encoding='utf-8')
        for path in sorted((REPO_ROOT / 'governance').glob('*.py'))
    )
    workflow_source = '\n'.join(
        path.read_text(encoding='utf-8') for path in sorted(WORKFLOWS_DIR.glob('*.yml'))
    )
    for key in sorted(_section('layout')):
        assert key in governance_source or key in workflow_source, (
            f'layout.{key} is configured but nothing reads it'
        )


def test_config_defaults_reproduce_the_previous_constants() -> None:
    """The shipped defaults must equal the values that were hardcoded.

    This slice's whole claim is that behaviour here does not change. Only a
    test pinning the defaults can hold that claim, so it is pinned against
    the literal previous constants rather than against the config file --
    which would just be asserting the file equals itself.
    """
    gates = _gates()
    assert gates['file_size_balance']['max_ratio'] == 15.894736842105264
    assert gates['file_size_balance']['min_files'] == 3
    assert gates['test_fallbacks']['excludes'] == []
    assert gates['module_docstrings']['excludes'] == []
    assert gates['module_docstrings']['exempt_below_significant_lines'] == 0
    assert gates['module_docstrings']['exempt_filename_matching_single_symbol'] is False
    assert gates['test_code_ratio']['min'] == 0.2072265302755321
    assert gates['test_code_ratio']['max'] == 2.00
    assert gates['test_code_ratio']['min_source_sloc'] == 50
    assert gates['coverage']['diff_floor'] == 80.0
    assert gates['coverage']['min_statements_for_track'] == 50
    assert gates['coverage']['min_branches_for_track'] == 20
    assert gates['coverage']['track_slack'] == 2
    assert gates['runtime_budget']['slowest_tests_limit'] == 10
    assert gates['packaging']['pyroma_min'] == 9
    assert gates['docstrings']['forbidden_title_verbs'] == [
        'calculate', 'generate', 'make', 'build',
    ]
    assert _int(_section('slice'), 'max_closing_references') == 2
    assert _section('changelog')['header'] == '# v{version}'
    assert _section('changelog')['newest'] == 'first'
    assert _str_list(_section('layout'), 'excludes') == ['__pycache__', 'build', 'dist']
