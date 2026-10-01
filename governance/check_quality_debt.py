#!/usr/bin/env python3
"""Run strict Ruff and dead-code checks, ratcheting evidenced inherited findings."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import vulture
from _common import REPO_ROOT, gate_config, resolve_package_dir
from _quality import BASELINE, baseline_ratchet, finding, new_findings, strict_policy_ratchet


def tool_findings() -> list[dict[str, str]]:
    """Collect complete strict lint and dead-code findings without soft failures."""
    package = resolve_package_dir('QUALITY DEBT GATE')
    config = gate_config('lint', 'QUALITY DEBT GATE')
    findings = []
    if config.get('ruff', True):
        result = subprocess.run(
            [sys.executable, '-m', 'ruff', 'check', str(package), 'governance', 'tests',
             '--config', 'governance/ruff.toml', '--output-format=json'], cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        )
        if result.returncode not in (0, 1):
            raise SystemExit(result.stderr or 'Ruff could not run')
        for item in json.loads(result.stdout):
            findings.append(finding(
                'ruff', Path(item['filename']), f'{item["code"]}: {item["message"]}',
                item['location']['row'],
            ))
    if config.get('dead_code', True):
        scanner = vulture.Vulture()
        scanner.scavenge([str(package)])
        for item in scanner.get_unused_code(min_confidence=80):
            findings.append(finding('vulture', item.filename, item.message, item.first_lineno))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-ref')
    args = parser.parse_args()
    failures = baseline_ratchet(args.base_ref) if args.base_ref else []
    if args.base_ref:
        failures.extend(strict_policy_ratchet(args.base_ref))
    current = tool_findings()
    new = new_findings(current)
    failures.extend(f'{item["path"]}: {item["message"]}' for item in new)
    if failures:
        print('QUALITY DEBT GATE -- FAIL', file=sys.stderr)
        print('\n'.join(f'  {message}' for message in failures), file=sys.stderr)
        return 1
    print(f'QUALITY DEBT GATE -- PASS ({len(current)} inherited findings; baseline={BASELINE.name})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
