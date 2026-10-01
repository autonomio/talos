#!/usr/bin/env python3
"""Privileged post-merge ruleset audit for full live parity on main.

Unlike the PR-time `ruleset_gate` (whose unprivileged token may not observe
`bypass_actors`), this runs on push to `main` with a privileged token and
REQUIRES `bypass_actors` to be present and equal to the snapshot. It is a
post-merge alarm, not a merge gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ruleset_gate as shared_ruleset_gate

LIVE_PAYLOAD_SNAPSHOT = 'live_ruleset.json'


def fail(message: str, *, code: int) -> int:
    print(f'privileged_ruleset_audit: {message}', file=sys.stderr)
    return code


def _write_live_payload_snapshot(output_dir: Path, payload: dict[str, object]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / LIVE_PAYLOAD_SNAPSHOT
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return path


def _privileged_live_ruleset_error(payload: dict[str, object]) -> tuple[str, int] | None:
    live_fields = set(payload)
    unexpected = (
        live_fields
        - shared_ruleset_gate.REQUIRED_TOP_LEVEL_FIELDS
        - shared_ruleset_gate.OPTIONAL_LIVE_TOP_LEVEL_FIELDS
        - shared_ruleset_gate.IGNORED_LIVE_FIELDS
    )
    if unexpected:
        return f'unexpected live ruleset field(s): {sorted(unexpected)}', 1

    missing_required = shared_ruleset_gate.REQUIRED_TOP_LEVEL_FIELDS - live_fields
    if missing_required:
        return f'expected live ruleset field(s) missing: {sorted(missing_required)}', 1

    missing_optional = shared_ruleset_gate.OPTIONAL_LIVE_TOP_LEVEL_FIELDS - live_fields
    if missing_optional:
        return (
            'privileged live ruleset missing required observable field(s): '
            f'{sorted(missing_optional)}',
            2,
        )

    return None


def normalize_privileged_live_ruleset(payload: dict[str, object]) -> dict[str, object]:
    comparable_fields = (
        shared_ruleset_gate.REQUIRED_TOP_LEVEL_FIELDS
        | shared_ruleset_gate.OPTIONAL_LIVE_TOP_LEVEL_FIELDS
    )
    return {key: payload[key] for key in sorted(comparable_fields)}


def run_audit(
    *,
    ruleset_file: str,
    repo: str | None,
    ruleset_id: int,
    output_dir: str,
    live_json: str | None = None,
) -> int:
    expected_ruleset = shared_ruleset_gate.normalize_snapshot_ruleset(
        shared_ruleset_gate.load_json_file(Path(ruleset_file))
    )
    live_payload = shared_ruleset_gate.load_live_ruleset(
        repo=repo,
        ruleset_id=str(ruleset_id),
        live_json=live_json,
    )
    snapshot_dir = Path(output_dir)

    live_error = _privileged_live_ruleset_error(live_payload)
    if live_error is not None:
        message, code = live_error
        _write_live_payload_snapshot(snapshot_dir, live_payload)
        return fail(message, code=code)

    live_ruleset = normalize_privileged_live_ruleset(live_payload)

    expected_comparable = {
        key: expected_ruleset[key]
        for key in sorted(live_ruleset)
    }

    if live_ruleset != expected_comparable:
        snapshot_path = _write_live_payload_snapshot(snapshot_dir, live_payload)
        print('privileged_ruleset_audit: ruleset drift detected', file=sys.stderr)
        print('expected:', file=sys.stderr)
        print(
            json.dumps(expected_comparable, indent=2, sort_keys=True),
            file=sys.stderr,
        )
        print('live:', file=sys.stderr)
        print(
            json.dumps(live_ruleset, indent=2, sort_keys=True),
            file=sys.stderr,
        )
        print(
            'privileged_ruleset_audit: wrote compared live payload to '
            f'{snapshot_path}',
            file=sys.stderr,
        )
        return 1

    print('PRIVILEGED RULESET AUDIT -- PASS')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='Privileged ruleset audit')
    parser.add_argument('--ruleset-file', required=True)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--ruleset-id', required=True, type=int)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    return run_audit(
        ruleset_file=args.ruleset_file,
        repo=args.repo,
        ruleset_id=args.ruleset_id,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    sys.exit(main())
