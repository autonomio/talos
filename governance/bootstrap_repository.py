#!/usr/bin/env python3
"""Configure a repository created from the law template."""

from __future__ import annotations

import argparse
import json
import keyword
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
from pathlib import Path
from typing import Final

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


# The bootstrap is the rename engine: it deliberately imports nothing from the
# governance/_common helper module (it keeps its own REPO_ROOT and
# _significant_lines below) so it stays a self-contained script that can run on
# a repository mid-specialization without depending on sibling gate modules.
# That is why the tomllib/tomli guard is inlined above rather than taken from
# _common.loads_toml, which every other TOML reader here goes through: routing
# this module through _common would trade self-containment for deduplication,
# and self-containment is the property that lets bootstrap run at all.
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
BOOTSTRAP_SCRIPT: Final[Path] = Path(__file__).resolve()
RULESET_PATH: Final[Path] = REPO_ROOT / '.github' / 'rulesets' / 'master.json'
TEMPLATE_LABEL_REPOSITORY: Final[str] = 'autonomio/talos'

TEXT_SUFFIXES: Final[frozenset[str]] = frozenset({
    '',
    '.cff',
    '.cfg',
    '.gitignore',
    '.ini',
    '.json',
    '.md',
    '.py',
    '.toml',
    '.txt',
    '.yaml',
    '.yml',
})

SKIP_DIRS: Final[frozenset[str]] = frozenset({
    '.git',
    '.mypy_cache',
    '.pytest_cache',
    '.ruff_cache',
    '.venv',
    '__pycache__',
    'build',
    'dist',
    'htmlcov',
    'node_modules',
    'venv',
})

KNOWN_SEED_PACKAGES: Final[frozenset[str]] = frozenset({
    'new_repository_template',
})


def _load_pyproject() -> dict[str, object]:
    path = REPO_ROOT / 'pyproject.toml'
    try:
        return tomllib.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return {}
    except tomllib.TOMLDecodeError as exc:
        raise SystemExit(f'bootstrap: cannot parse pyproject.toml: {exc}') from exc


def _slug_to_package(slug: str) -> str:
    package = re.sub(r'[^0-9A-Za-z_]+', '_', slug).strip('_').lower()
    package = re.sub(r'_+', '_', package)
    if not package:
        raise SystemExit('bootstrap: repository name did not produce a package name')
    if package[0].isdigit():
        package = f'_{package}'
    if keyword.iskeyword(package):
        package = f'{package}_pkg'
    return package


def _repo_name_from_env() -> str:
    repository = os.environ.get('GITHUB_REPOSITORY', '')
    if '/' in repository:
        return repository.rsplit('/', 1)[1]
    return Path.cwd().name


def _owner_from_env() -> str | None:
    repository = os.environ.get('GITHUB_REPOSITORY', '')
    if '/' in repository:
        return repository.split('/', 1)[0]
    return os.environ.get('GITHUB_REPOSITORY_OWNER') or None


def _package_roots_from_config(pyproject: dict[str, object]) -> set[str]:
    roots = set(KNOWN_SEED_PACKAGES)
    project = pyproject.get('project')
    if isinstance(project, dict):
        name = project.get('name')
        if isinstance(name, str) and name:
            roots.add(_slug_to_package(name))

    tool = pyproject.get('tool')
    if isinstance(tool, dict):
        pyright = tool.get('pyright')
        if isinstance(pyright, dict):
            include = pyright.get('include')
            if isinstance(include, list):
                roots.update(str(item) for item in include if isinstance(item, str))

    config_path = REPO_ROOT / 'governance.yml'
    if config_path.is_file():
        import yaml
        layout = yaml.safe_load(config_path.read_text(encoding='utf-8')).get('layout', {})
        if isinstance(layout, dict):
            root = layout.get('package_root')
            if isinstance(root, str) and root:
                roots.add(root)

    return {root for root in roots if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', root)}


def _iter_text_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted(REPO_ROOT.rglob('*')):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(REPO_ROOT).parts):
            continue
        if path == BOOTSTRAP_SCRIPT:
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name not in {'LICENSE', 'README'}:
            continue
        files.append(path)
    return files


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return None
    except UnicodeDecodeError:
        return None


def _write_text_if_changed(path: Path, text: str) -> bool:
    current = _read_text(path)
    if current == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return True


def _significant_lines(path: Path) -> int:
    count = 0
    for line in path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            count += 1
    return count


def _replace_identity_tokens(
    repo_slug: str,
    package_name: str,
    old_packages: set[str],
    owner: str | None,
) -> int:
    changed = 0
    display_name = repo_slug.replace('-', ' ').replace('_', ' ').title()
    description = 'Python package with repository law built in.'
    dashed_seeds = {pkg.replace('_', '-') for pkg in old_packages}
    label_template_sentinel = '__REPOSITORY_LAW_LABEL_TEMPLATE__'
    owner_name = owner or 'Autonomio'

    for path in _iter_text_files():
        text = _read_text(path)
        if text is None:
            continue
        rel_path = path.relative_to(REPO_ROOT).as_posix()
        if rel_path.startswith('.github/workflows/'):
            continue
        # References to the template's own slug -- provenance in README, the
        # SETUP runbook, the bootstrap label source -- must survive
        # specialization unchanged, in every file, not just this script.
        updated = text.replace('autonomio/talos', label_template_sentinel)
        for old_package in sorted(old_packages, key=len, reverse=True):
            if old_package != package_name:
                updated = updated.replace(old_package, package_name)
        for old_slug in sorted(dashed_seeds, key=len, reverse=True):
            if old_slug != repo_slug:
                updated = updated.replace(old_slug, repo_slug)
        updated = updated.replace('{REPOSITORY_NAME}', repo_slug)
        updated = updated.replace('{REPOSITORY_OWNER}', owner_name)
        updated = updated.replace('{ONE_SENTENCE_DESCRIPTION}', description)
        updated = updated.replace('{DISPLAY_NAME}', display_name)
        updated = updated.replace(f'Autonomio/{package_name}', f'{owner_name}/{repo_slug}')
        updated = updated.replace(label_template_sentinel, 'autonomio/talos')
        if _write_text_if_changed(path, updated):
            changed += 1
    return changed


def _replace_line(text: str, pattern: str, replacement: str) -> str:
    return re.sub(pattern, replacement, text, flags=re.MULTILINE)


def _rewrite_pyproject(repo_slug: str, package_name: str) -> bool:
    path = REPO_ROOT / 'pyproject.toml'
    text = path.read_text(encoding='utf-8')
    text = _replace_line(text, r'^name = ".*"$', f'name = "{repo_slug}"')
    text = _replace_line(
        text,
        r'^include = \["[A-Za-z_][A-Za-z0-9_]*\*"\]$',
        f'include = ["{package_name}*"]',
    )
    text = _replace_line(
        text,
        r'^include = \["[A-Za-z_][A-Za-z0-9_]*"\]$',
        f'include = ["{package_name}"]',
    )
    text = _replace_line(
        text,
        r'^known-first-party = \["[A-Za-z_][A-Za-z0-9_]*"\]$',
        f'known-first-party = ["{package_name}"]',
    )
    text = _replace_line(
        text,
        r'^paths = \["[A-Za-z_][A-Za-z0-9_]*"\]$',
        f'paths = ["{package_name}"]',
    )
    text = _replace_line(
        text,
        r'^source = \["[A-Za-z_][A-Za-z0-9_]*"\]$',
        f'source = ["{package_name}"]',
    )
    text = re.sub(r'\n{3,}', '\n\n', text).rstrip() + '\n'
    return _write_text_if_changed(path, text)


def _create_package_baseline(repo_slug: str, package_name: str) -> int:
    changed = 0
    package_dir = REPO_ROOT / package_name
    package_dir.mkdir(exist_ok=True)
    init_path = package_dir / '__init__.py'
    if not init_path.exists():
        changed += _write_text_if_changed(
            init_path,
            f'"""Public package surface for {repo_slug}."""\n\n__all__: list[str] = []\n',
        )

    package_tests = REPO_ROOT / 'tests' / 'package' / 'test_import.py'
    if not package_tests.exists():
        changed += _write_text_if_changed(
            package_tests,
            (
                'from __future__ import annotations\n\n'
                'import importlib\n\n\n'
                'def test_package_imports() -> None:\n'
                f"    assert importlib.import_module('{package_name}').__name__ == '{package_name}'\n"
            ),
        )

    changelog = REPO_ROOT / 'CHANGELOG.md'
    if not changelog.exists():
        changed += _write_text_if_changed(
            changelog,
            '# v0.1.0\n\n- Initial repository law baseline.\n',
        )

    return changed


def _typing_budget() -> dict[str, object]:
    return {
        'patterns': {
            'any_annotation': {'pattern': r':\s*Any\b', 'total': 0},
            'any_return': {'pattern': r'->\s*Any\b', 'total': 0},
            'any_import': {'pattern': r'from typing import[^\n]*\bAny\b', 'total': 0},
            'cast_any': {'pattern': r'cast\([^)]*\bAny\b', 'total': 0},
            'dict_any': {'pattern': r'dict\[[^]]*\bAny\b', 'total': 0},
            'list_any': {'pattern': r'list\[\s*Any\b', 'total': 0},
            'tuple_any': {'pattern': r'tuple\[[^]]*\bAny\b', 'total': 0},
            'type_ignore': {'pattern': r'#\s*type:\s*ignore', 'total': 0},
            'pyright_ignore': {'pattern': r'#\s*pyright:\s*ignore', 'total': 0},
            'noqa': {'pattern': r'#\s*noqa', 'total': 0},
        },
        'any_references': {'total': 0},
        'pyright_errors': {'total': 0},
        'pyright_warnings': {'total': 0},
    }


def _fail_loud_budget() -> dict[str, object]:
    categories = [
        'bare_except',
        'empty_pass',
        'empty_ellipsis',
        'empty_return_none',
        'empty_continue_break',
        'contextlib_suppress',
        'errors_ignore_kwarg',
    ]
    return {'categories': {category: {'total': 0} for category in categories}}


def _existing_module_budgets() -> dict[str, int]:
    """The module budgets this repository already declares, if any."""
    path = REPO_ROOT / '.github' / 'budgets.json'
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}
    modules = data.get('modules') if isinstance(data, dict) else None
    if not isinstance(modules, dict):
        return {}
    return {str(k): v for k, v in modules.items() if isinstance(v, int)}


def _module_budgets(package_name: str) -> dict[str, int]:
    payload: dict[str, int] = {}
    package_dir = REPO_ROOT / package_name
    if package_dir.is_dir():
        for source_file in sorted(package_dir.rglob('*.py')):
            rel = source_file.relative_to(REPO_ROOT).as_posix()
            payload[rel] = max(10, _significant_lines(source_file) + 20)
    governance_dir = REPO_ROOT / 'governance'
    if governance_dir.is_dir():
        if (governance_dir / '__init__.py').is_file():
            payload['governance/__init__.py'] = 10
        # The template's governance budgets are already calibrated -- several
        # are deliberately tighter than a measurement would give. Carry them
        # over and measure only modules that have none, rather than re-deriving
        # every number: a second, re-derived copy is what drifted from the
        # first and left every derived repository's opening PR failing the
        # module-budget gate.
        existing = _existing_module_budgets()
        for module in sorted(governance_dir.glob('*.py')):
            rel = module.relative_to(REPO_ROOT).as_posix()
            payload[rel] = existing.get(rel, max(10, _significant_lines(module) + 20))
    return payload


def _write_budgets(package_name: str) -> int:
    # One file, one write. The sections that used to be separate files are
    # composed here so a bootstrap cannot leave them half-rewritten.
    path = REPO_ROOT / '.github' / 'budgets.json'
    existing = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    payload = {
        'schema_version': 1,
        'typing': _typing_budget(),
        'fail_loud': _fail_loud_budget(),
        'modules': _module_budgets(package_name),
        'coverage': existing.get('coverage', {'line': 0, 'branch': 0}),
        'runtime': existing.get('runtime', {'max_total_seconds': 300}),
    }
    return _write_text_if_changed(path, json.dumps(payload, indent=2) + '\n')


def _is_baseline_seed_package_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    files = [
        p
        for p in path.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts
    ]
    if len(files) != 1 or files[0].name != '__init__.py':
        return False
    text = files[0].read_text(encoding='utf-8')
    return (
        text.startswith('"""Public package surface for ')
        and '__all__: list[str] = []' in text
    )


def _rename_seed_package_dirs(package_name: str, old_packages: set[str]) -> int:
    changed = 0
    target_exists = (REPO_ROOT / package_name).exists()
    for old_package in sorted(old_packages):
        old_dir = REPO_ROOT / old_package
        if old_package == package_name or not old_dir.is_dir():
            continue
        if not target_exists:
            old_dir.rename(REPO_ROOT / package_name)
            target_exists = True
            changed += 1
            continue
        if _is_baseline_seed_package_dir(old_dir):
            shutil.rmtree(old_dir)
            changed += 1
    return changed


CODEQL_CONTEXT: Final[str] = 'PR Checks CodeQL (python)'


def _drop_codeql_law(text: str) -> str:
    # Remove the CodeQL law from "## The laws", renumber the remaining laws
    # sequentially, and correct the count header. The laws <-> ruleset
    # bijection (pr_checks_honesty) requires the CodeQL law and the CodeQL
    # required-status-check to be dropped together.
    out: list[str] = []
    in_laws = False
    counter = 0
    for line in text.split('\n'):
        if line.startswith('## '):
            in_laws = line.strip() == '## The laws'
            out.append(line)
            continue
        if in_laws and re.match(r'^\d+\.\s', line):
            if f'*({CODEQL_CONTEXT})*' in line:
                continue
            counter += 1
            line = re.sub(r'^\d+\.', f'{counter}.', line)
        out.append(line)
    return '\n'.join(out).replace(
        'Eleven laws. Ten are workflow gates on every PR; the eleventh',
        'Ten laws. Nine are workflow gates on every PR; the tenth',
    )


def _drop_codeql_context(ruleset: dict[str, object]) -> bool:
    rules = ruleset.get('rules')
    if not isinstance(rules, list):
        return False
    changed = False
    for rule in rules[:]:
        if not isinstance(rule, dict):
            continue
        kind = rule.get('type')
        params = rule.get('parameters')
        if kind not in ('required_status_checks', 'code_scanning') or not isinstance(params, dict):
            continue
        key = 'code_scanning_tools' if kind == 'code_scanning' else 'required_status_checks'
        entries = params.get(key)
        if not isinstance(entries, list):
            continue
        field, target = ('tool', 'CodeQL') if kind == 'code_scanning' else ('context', CODEQL_CONTEXT)
        kept = [entry for entry in entries if not (isinstance(entry, dict) and entry.get(field) == target)]
        if len(kept) != len(entries):
            params[key] = kept
            if not kept and kind == 'code_scanning':
                rules.remove(rule)
            changed = True
    return changed


def _drop_codeql_from_config(text: str) -> str:
    # Remove the whole `gates.codeql` block from governance.yml, preserving
    # the rest of the file's formatting. governance.yml is the contract anchor
    # the config tests pin the ruleset to, so it must lose CodeQL alongside
    # the ruleset, the laws, and the workflow. The block is dropped entire
    # rather than just its `context:` line: a gate left enabled with no
    # context would still be owed a required check that no longer exists.
    lines = text.splitlines(keepends=True)
    kept: list[str] = []
    dropping = False
    for line in lines:
        stripped = line.strip()
        if re.match(r'^  codeql:\s*$', line.rstrip('\n')):
            dropping = True
            continue
        if dropping:
            # Blank lines and deeper-indented keys belong to the block; the
            # next sibling key (two-space indent) ends it.
            if not stripped or line.startswith('    '):
                continue
            dropping = False
        kept.append(line)
    return ''.join(kept)


def disable_codeql(repo_root: Path = REPO_ROOT) -> int:
    """Remove CodeQL laws, analysis, findings requirements and mirrored config.

    Unsupported targets retain other scanner requirements and remaining laws;
    the ruleset fixtures follow the same removal.
    """
    changed = 0
    laws_path = repo_root / 'CLAUDE.md'
    if laws_path.is_file():
        changed += _write_text_if_changed(
            laws_path, _drop_codeql_law(laws_path.read_text(encoding='utf-8'))
        )
    ruleset_path = repo_root / '.github' / 'rulesets' / 'master.json'
    if ruleset_path.is_file():
        ruleset = json.loads(ruleset_path.read_text(encoding='utf-8'))
        if isinstance(ruleset, dict) and _drop_codeql_context(ruleset):
            changed += _write_text_if_changed(
                ruleset_path, json.dumps(ruleset, indent=2) + '\n'
            )
    workflow_path = repo_root / '.github' / 'workflows' / 'pr_checks_codeql.yml'
    if workflow_path.is_file():
        workflow_path.unlink()
        changed += 1
    # Keep the ruleset test fixtures consistent with the de-CodeQL'd snapshot,
    # so the ruleset-gate and audit contract tests do not see false drift.
    fixtures_dir = repo_root / 'governance' / 'tests' / 'fixtures' / 'github'
    if fixtures_dir.is_dir():
        for fixture in sorted(fixtures_dir.glob('*.json')):
            data = json.loads(fixture.read_text(encoding='utf-8'))
            if isinstance(data, dict) and _drop_codeql_context(data):
                changed += _write_text_if_changed(
                    fixture, json.dumps(data, indent=2) + '\n'
                )
    # governance.yml is the contract anchor the config tests pin the ruleset
    # to; it must lose CodeQL too, or test_governance_config fails the private
    # bootstrap PR (the ruleset drops CodeQL while the config still lists it).
    config_path = repo_root / 'governance.yml'
    if config_path.is_file():
        changed += _write_text_if_changed(
            config_path, _drop_codeql_from_config(config_path.read_text(encoding='utf-8'))
        )
    return changed


def _apply_file_bootstrap(
    repo_slug: str,
    package_name: str,
    owner: str | None,
    codeql: str = 'supported',
) -> None:
    # Idempotency. The bootstrap workflow triggers on every push to main,
    # but the rename + budget rewrite must run exactly once: once the seed
    # package is gone the repository is already specialized, and re-deriving
    # the module budgets would revert the values later slices have tuned,
    # opening a spurious bootstrap PR on every merge. Skip that work when
    # already specialized -- but still reconcile CodeQL, which is itself
    # idempotent and must be able to run if the repository lost CodeQL
    # support (e.g. went private without Advanced Security) after the
    # initial bootstrap.
    changed = 0
    if any((REPO_ROOT / seed).is_dir() for seed in KNOWN_SEED_PACKAGES):
        pyproject = _load_pyproject()
        old_packages = _package_roots_from_config(pyproject)
        old_packages.add(package_name)
        changed += _rename_seed_package_dirs(package_name, old_packages)
        changed += _replace_identity_tokens(repo_slug, package_name, old_packages, owner)
        changed += _rewrite_pyproject(repo_slug, package_name)
        changed += _create_package_baseline(repo_slug, package_name)
        changed += _write_budgets(package_name)
    else:
        print('bootstrap: repository already specialized; skipping rename and budget rewrite.')
    if codeql == 'unsupported':
        changed += disable_codeql()
    print(f'bootstrap: file bootstrap complete ({changed} file groups changed)')


def _run_gh(args: list[str], *, input_text: str | None = None) -> str:
    if shutil.which('gh') is None:
        raise SystemExit('bootstrap: gh is required for GitHub configuration')
    result = subprocess.run(
        ['gh', *args],
        input=input_text,
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or 'no output'
        raise SystemExit(
            f"bootstrap: gh {' '.join(args)} failed with exit {result.returncode}: {stderr}"
        )
    return result.stdout


def _repo_full_name(owner: str | None, repo_slug: str) -> str:
    if owner:
        return f'{owner}/{repo_slug}'
    repository = os.environ.get('GITHUB_REPOSITORY', '')
    if '/' in repository:
        return repository
    raise SystemExit('bootstrap: --owner is required outside GitHub Actions for --github-only')


def _find_ruleset_id(repo: str, ruleset_name: str) -> int | None:
    output = _run_gh(['api', '--paginate', f'repos/{repo}/rulesets'])
    payload = json.loads(output)
    if not isinstance(payload, list):
        raise SystemExit('bootstrap: expected list from repository rulesets API')
    for item in payload:
        if (
            isinstance(item, dict)
            and item.get('name') == ruleset_name
            and item.get('source_type') == 'Repository'
        ):
            ruleset_id = item.get('id')
            if isinstance(ruleset_id, int):
                return ruleset_id
    return None


def _apply_ruleset(repo: str) -> int:
    ruleset = json.loads(RULESET_PATH.read_text(encoding='utf-8'))
    name = ruleset.get('name')
    if not isinstance(name, str) or not name:
        raise SystemExit('bootstrap: ruleset snapshot must have a name')
    ruleset_id = _find_ruleset_id(repo, name)
    body = json.dumps(ruleset)
    if ruleset_id is None:
        output = _run_gh(['api', '-X', 'POST', f'repos/{repo}/rulesets', '--input', '-'], input_text=body)
        created = json.loads(output)
        ruleset_id = int(created['id'])
        print(f'bootstrap: created ruleset {name!r} -> {ruleset_id}')
    else:
        _run_gh(['api', '-X', 'PUT', f'repos/{repo}/rulesets/{ruleset_id}', '--input', '-'], input_text=body)
        print(f'bootstrap: updated ruleset {name!r} -> {ruleset_id}')
    _run_gh(['variable', 'set', 'RULESET_ID', '--repo', repo, '--body', str(ruleset_id)])
    print('bootstrap: set repository variable RULESET_ID')
    return ruleset_id


def _copy_labels(repo: str, template_repo: str) -> None:
    existing = _run_gh(['api', '--paginate', f'repos/{repo}/labels', '--jq', '.[].name'])
    for label in existing.splitlines():
        if not label:
            continue
        encoded_label = urllib.parse.quote(label, safe='')
        _run_gh(['api', '-X', 'DELETE', f'repos/{repo}/labels/{encoded_label}'])

    labels_raw = _run_gh(['api', '--paginate', f'repos/{template_repo}/labels'])
    labels = json.loads(labels_raw)
    if not isinstance(labels, list):
        raise SystemExit('bootstrap: expected list from label template repository')
    for label in labels:
        if not isinstance(label, dict):
            continue
        name = str(label.get('name', ''))
        color = str(label.get('color', ''))
        description = str(label.get('description') or '')
        if not name or not color:
            continue
        _run_gh([
            'api',
            '-X',
            'POST',
            f'repos/{repo}/labels',
            '-f',
            f'name={name}',
            '-f',
            f'color={color}',
            '-f',
            f'description={description}',
        ])
    print(f'bootstrap: copied labels from {template_repo}')


def _apply_github_bootstrap(repo_slug: str, owner: str | None) -> None:
    if not (os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN')):
        raise SystemExit(
            'bootstrap: GH_TOKEN or GITHUB_TOKEN is required for --github-only'
        )
    repo = _repo_full_name(owner, repo_slug)
    subprocess.run(
        [sys.executable, str(REPO_ROOT / 'scripts' / 'configure_repository.py'),
         '--repo', repo, '--apply'], check=True, cwd=REPO_ROOT,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-name', default=_repo_name_from_env())
    parser.add_argument('--owner', default=_owner_from_env())
    parser.add_argument('--package-name')
    parser.add_argument('--files-only', action='store_true')
    parser.add_argument('--github-only', action='store_true')
    parser.add_argument('--codeql', choices=('supported', 'unsupported'), default='supported')
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_slug = str(args.repo_name).strip()
    if not repo_slug:
        raise SystemExit('bootstrap: --repo-name cannot be empty')
    package_name = str(args.package_name or _slug_to_package(repo_slug)).strip()
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', package_name):
        raise SystemExit(
            f'bootstrap: package name {package_name!r} is not a valid Python identifier'
        )

    if args.files_only and args.github_only:
        raise SystemExit('bootstrap: choose at most one of --files-only and --github-only')

    if not args.github_only:
        _apply_file_bootstrap(repo_slug, package_name, args.owner, codeql=args.codeql)
    if not args.files_only:
        _apply_github_bootstrap(repo_slug, args.owner)
    return 0


if __name__ == '__main__':
    sys.exit(main())
