#!/usr/bin/env python3
"""Bind shared CI coverage to the checkout, workflow run and locked environment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / 'coverage-evidence.json'


def binding(attempt: str | None = None) -> dict[str, str]:
    """Compute the exact producer/consumer identity; absent evidence fails loudly."""
    source = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
    ).strip()
    result = {'source': source, 'coverage_sha256': hashlib.sha256(
        (ROOT / 'coverage.json').read_bytes(),
    ).hexdigest()}
    for name in ('GITHUB_SHA', 'GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT'):
        result[name] = os.environ[name]
    if attempt is not None:
        result['GITHUB_RUN_ATTEMPT'] = attempt
    if source != result['GITHUB_SHA']:
        raise ValueError('coverage checkout differs from the workflow source')
    for name in ('dev-env.txt', 'runtime-env.txt', 'build-tools.txt'):
        result[name] = hashlib.sha256((ROOT / 'requirements/ci' / name).read_bytes()).hexdigest()
    return result


def verify() -> None:
    """Reject stale, tampered, missing or foreign test-run coverage."""
    if json.loads(RECEIPT.read_text(encoding='utf-8')) != binding(os.environ['TEST_PRODUCER_ATTEMPT']):
        raise ValueError('coverage evidence differs from this source, run, attempt or lock set')


def main() -> int:
    """Write after successful tests, or verify before coverage gates consume it."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.write:
        RECEIPT.write_text(json.dumps(binding(), indent=2) + '\n', encoding='utf-8')
    else:
        verify()
    print('Coverage evidence matches this source and workflow run.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
