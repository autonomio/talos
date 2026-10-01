#!/usr/bin/env python3
"""Preview or explicitly apply the reviewed Talos labels and master ruleset."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
REPOSITORY: Final[str] = 'autonomio/talos'


def gh(*args: str, body: str | None = None) -> str:
    """Run GitHub CLI without suppressing configuration failures."""
    result = subprocess.run(['gh', *args], input=body, text=True, capture_output=True, check=True)
    return result.stdout


def configure_labels(repo: str, labels: list[dict[str, str]]) -> None:
    """Upsert declared labels while preserving every unrelated existing label."""
    for label in labels:
        gh('label', 'create', label['name'], '--repo', repo, '--force',
           '--color', label['color'], '--description', label['description'])


def configure_ruleset(repo: str, snapshot: dict[str, object]) -> None:
    """Reconcile the repository-owned ruleset and record its live identifier."""
    existing = json.loads(gh('api', '--paginate', f'repos/{repo}/rulesets'))
    matches = [item for item in existing if item.get('source_type') == 'Repository'
               and item.get('name') == snapshot['name']]
    if len(matches) > 1:
        raise ValueError('multiple repository rulesets have the declared name')
    body = json.dumps(snapshot)
    endpoint = f'repos/{repo}/rulesets'
    if matches:
        endpoint += f"/{matches[0]['id']}"
    applied = json.loads(gh('api', '-X', 'PUT' if matches else 'POST', endpoint,
                            '--input', '-', body=body))
    gh('variable', 'set', 'RULESET_ID', '--repo', repo, '--body', str(applied['id']))
    print(f"CONFIGURE -- PASS: {snapshot['name']} ({applied['id']})")


def main() -> int:
    """Keep configuration read-only unless the operator supplies --apply."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default=REPOSITORY)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--labels-only', action='store_true')
    args = parser.parse_args()
    if args.repo != REPOSITORY:
        parser.error(f'this configuration belongs to {REPOSITORY}')
    labels = json.loads((REPO_ROOT / '.github/labels.json').read_text(encoding='utf-8'))
    for label in labels:
        if set(label) != {'name', 'color', 'description'} or not all(
            isinstance(value, str) and value for value in label.values()
        ):
            parser.error('each declared label must have name, color and description')
    snapshot = json.loads((REPO_ROOT / '.github/rulesets/master.json').read_text(encoding='utf-8'))
    branches = snapshot['conditions']['ref_name']['include']
    if branches != ['refs/heads/master']:
        parser.error('the reviewed ruleset must target master explicitly')
    if not args.apply:
        print(json.dumps({'repository': args.repo, 'labels': labels,
                          'ruleset': None if args.labels_only else snapshot}, indent=2))
        return 0
    if not os.environ.get('GH_TOKEN'):
        parser.error('GH_TOKEN is required for explicit configuration application')
    configure_labels(args.repo, labels)
    if not args.labels_only:
        configure_ruleset(args.repo, snapshot)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
