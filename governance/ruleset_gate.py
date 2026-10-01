#!/usr/bin/env python3
"""Ruleset drift gate for the protected main-branch ruleset."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

from _common import exit_if_disabled

BANNER: Final[str] = 'RULESET GATE'

REQUIRED_TOP_LEVEL_FIELDS: Final[frozenset[str]] = frozenset({
    'name',
    'target',
    'enforcement',
    'conditions',
    'rules',
})

OPTIONAL_LIVE_TOP_LEVEL_FIELDS: Final[frozenset[str]] = frozenset({
    'bypass_actors',
})

SNAPSHOT_TOP_LEVEL_FIELDS: Final[frozenset[str]] = frozenset(
    REQUIRED_TOP_LEVEL_FIELDS | OPTIONAL_LIVE_TOP_LEVEL_FIELDS
)

IGNORED_LIVE_FIELDS: Final[frozenset[str]] = frozenset({
    '_links',
    'created_at',
    'current_user_can_bypass',
    'id',
    'node_id',
    'source',
    'source_type',
    'updated_at',
})


def fail(message: str, *, code: int) -> int:
    print(f'ruleset_gate: {message}', file=sys.stderr)
    return code


def load_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise SystemExit(fail(f'file not found: {path}', code=2)) from None
    except json.JSONDecodeError as exc:
        raise SystemExit(
            fail(f'invalid JSON in {path}: {exc}', code=2)
        ) from exc
    if not isinstance(payload, dict):
        raise SystemExit(
            fail(f'expected top-level JSON object in {path}', code=2)
        )
    return payload


def load_live_ruleset(
    *,
    repo: str | None,
    ruleset_id: str,
    live_json: str | None,
) -> dict[str, Any]:
    if live_json is not None:
        return load_json_file(Path(live_json))

    if not repo:
        raise SystemExit(
            fail('--repo is required unless --live-json is provided', code=2)
        )

    try:
        result = subprocess.run(
            ['gh', 'api', f'repos/{repo}/rulesets/{ruleset_id}'],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(fail(f'gh not found: {exc}', code=2)) from exc

    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or 'no error output'
        raise SystemExit(
            fail(
                f'gh api repos/{repo}/rulesets/{ruleset_id} failed '
                f'(exit {result.returncode}): {stderr}',
                code=2,
            )
        )

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            fail(f'live ruleset response is not valid JSON: {exc}', code=2)
        ) from exc

    if not isinstance(payload, dict):
        raise SystemExit(
            fail('live ruleset response is not a JSON object', code=2)
        )

    return payload


def normalize_snapshot_ruleset(payload: dict[str, Any]) -> dict[str, Any]:
    snapshot_fields = set(payload)
    unexpected = snapshot_fields - SNAPSHOT_TOP_LEVEL_FIELDS
    if unexpected:
        raise SystemExit(
            fail(
                f'unexpected snapshot ruleset field(s): {sorted(unexpected)}',
                code=2,
            )
        )

    missing = SNAPSHOT_TOP_LEVEL_FIELDS - snapshot_fields
    if missing:
        raise SystemExit(
            fail(
                f'expected snapshot ruleset field(s) missing: {sorted(missing)}',
                code=2,
            )
        )

    return payload


def normalize_live_ruleset(payload: dict[str, Any]) -> tuple[dict[str, Any], frozenset[str]]:
    live_fields = set(payload)
    unexpected = (
        live_fields
        - REQUIRED_TOP_LEVEL_FIELDS
        - OPTIONAL_LIVE_TOP_LEVEL_FIELDS
        - IGNORED_LIVE_FIELDS
    )
    if unexpected:
        raise SystemExit(
            fail(
                f'unexpected live ruleset field(s): {sorted(unexpected)}',
                code=1,
            )
        )

    missing = REQUIRED_TOP_LEVEL_FIELDS - live_fields
    if missing:
        raise SystemExit(
            fail(
                f'expected live ruleset field(s) missing: {sorted(missing)}',
                code=1,
            )
        )

    comparable_fields = REQUIRED_TOP_LEVEL_FIELDS | (
        OPTIONAL_LIVE_TOP_LEVEL_FIELDS & live_fields
    )

    missing_optional = OPTIONAL_LIVE_TOP_LEVEL_FIELDS - live_fields
    if missing_optional:
        print(
            'ruleset_gate: live ruleset omitted non-observable field(s): '
            f'{sorted(missing_optional)}; comparing observable subset only',
            file=sys.stderr,
        )

    return (
        {key: payload[key] for key in sorted(comparable_fields)},
        frozenset(comparable_fields),
    )


def main() -> int:
    exit_if_disabled('ruleset', BANNER)
    parser = argparse.ArgumentParser(description='Ruleset drift gate')
    parser.add_argument('--ruleset-file', required=True)
    parser.add_argument('--repo')
    parser.add_argument('--ruleset-id', required=True)
    parser.add_argument('--live-json')
    args = parser.parse_args()

    expected_ruleset = normalize_snapshot_ruleset(
        load_json_file(Path(args.ruleset_file))
    )
    live_ruleset, compared_fields = normalize_live_ruleset(
        load_live_ruleset(
            repo=args.repo,
            ruleset_id=args.ruleset_id,
            live_json=args.live_json,
        )
    )
    expected_ruleset = {
        key: expected_ruleset[key]
        for key in sorted(compared_fields)
    }

    if live_ruleset != expected_ruleset:
        print('ruleset_gate: ruleset drift detected', file=sys.stderr)
        print('expected:', file=sys.stderr)
        print(
            json.dumps(expected_ruleset, indent=2, sort_keys=True),
            file=sys.stderr,
        )
        print('live:', file=sys.stderr)
        print(
            json.dumps(live_ruleset, indent=2, sort_keys=True),
            file=sys.stderr,
        )
        return 1

    print('RULESET GATE -- PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
