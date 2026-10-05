#!/usr/bin/env python3
"""Select successful protected-master CI evidence before any release mutation."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from create_release import compute_tag, current_version

REPOSITORY = 'autonomio/talos'


def _record(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError('GitHub response must contain a JSON object with string keys')
    return dict(value)


def _api(path: str) -> dict[str, object]:
    result = subprocess.run(['gh', 'api', f'repos/{REPOSITORY}/{path}'],
                            capture_output=True, text=True, check=True)
    return _record(json.loads(result.stdout))


def validate_run(run: dict[str, object], source: str, workflow_id: object) -> int:
    """Reject failed, foreign, pull-request or differently sourced CI evidence."""
    expected = {'workflow_id': workflow_id, 'head_sha': source, 'head_branch': 'master',
                'event': 'push', 'status': 'completed', 'conclusion': 'success',
                'path': '.github/workflows/ci.yml'}
    for field, value in expected.items():
        if run.get(field) != value:
            raise ValueError(f'CI evidence disagrees on {field}: {run.get(field)!r}')
    if _record(run.get('head_repository')).get('full_name') != REPOSITORY:
        raise ValueError('CI evidence belongs to another repository')
    run_id = run.get('id')
    if type(run_id) is not int or run_id <= 0:
        raise ValueError('CI evidence has no valid run identifier')
    return run_id


def select_run(event_name: str, event: dict[str, object], source: str) -> dict[str, object]:
    """Resolve automatic completion or recovery to the canonical CI workflow."""
    if event_name == 'workflow_run':
        run_id = _record(event.get('workflow_run')).get('id')
        if type(run_id) is not int or run_id <= 0:
            raise ValueError('workflow_run event has no valid run identifier')
        return _api(f'actions/runs/{run_id}')
    if event_name != 'workflow_dispatch':
        raise ValueError('release requires successful CI completion or explicit recovery dispatch')
    result = _api(f'actions/workflows/ci.yml/runs?head_sha={source}&branch=master&event=push&status=success')
    runs = result.get('workflow_runs')
    if not isinstance(runs, list) or not runs:
        raise ValueError('selected master commit has no successful full CI run')
    return _record(runs[0])


def main() -> None:
    """Emit one immutable source/version/run selection, or skip a superseded head."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    if os.environ['GITHUB_REPOSITORY'] != REPOSITORY:
        parser.error('release repository must be autonomio/talos')
    if re.fullmatch(r'[0-9a-f]{40}', args.expected_sha) is None:
        parser.error('release source must be a full commit SHA')
    event = _record(json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text()))
    run = select_run(os.environ['GITHUB_EVENT_NAME'], event, args.expected_sha)
    workflow_id = _api('actions/workflows/ci.yml')['id']
    tested_source = run.get('head_sha')
    if not isinstance(tested_source, str):
        raise ValueError('CI evidence has no source commit')
    run_id = validate_run(run, tested_source, workflow_id)
    master = _record(_api('git/ref/heads/master')['object'])['sha']
    output = Path(os.environ['GITHUB_OUTPUT'])
    if tested_source != args.expected_sha or master != args.expected_sha:
        output.write_text('ready=false\n')
        print('RELEASE -- SKIP: completed CI head has been superseded on master')
        return
    tag = compute_tag(current_version())
    if args.tag and args.tag != tag:
        parser.error('recovery tag must match the selected source version')
    output.write_text(f'ready=true\ntag={tag}\nsource_sha={args.expected_sha}\nci_run_id={run_id}\n')
    print(f'RELEASE CANDIDATE: {tag} at {args.expected_sha}, CI {run_id}')


if __name__ == '__main__':
    main()
