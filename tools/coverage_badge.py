"""Publish verified statement coverage with its source and CI provenance."""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import re

from tools.check_statement_coverage import check_statement_coverage


def publish(report: Path, output: Path, commit: str, run_url: str) -> dict[str, object]:
    """Write a badge and report only after validating the complete package."""
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('source commit must be a full lowercase Git SHA')
    if not re.fullmatch(r'https://github.com/autonomio/talos/actions/runs/[1-9][0-9]*', run_url):
        raise ValueError('coverage must link to a Talos GitHub Actions run')
    root = Path(__file__).resolve().parents[1]
    covered, total = check_statement_coverage(report, root / 'talos')
    source = report.read_bytes()
    payload = json.loads(source)
    percent = f'{100 * covered / total:.2f}%'
    summary: dict[str, object] = {
        'schema_version': 1, 'scope': 'all Talos package statements',
        'source_commit': commit, 'ci_run': run_url,
        'report_sha256': hashlib.sha256(source).hexdigest(),
        'covered_statements': covered, 'total_statements': total,
        'statement_percent': 100 * covered / total,
        'modules': len(payload['files']),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'coverage.json').write_bytes(source)
    (output / 'coverage-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'coverage.svg').write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="236" height="28" '
        f'role="img" aria-label="Statement coverage: {percent}">'
        f'<title>Statement coverage: {percent}; source {commit}</title>'
        '<rect width="168" height="28" fill="#252D33"/>'
        '<rect x="168" width="68" height="28" fill="#2A4A70"/>'
        '<g fill="#F7F7F2" font-family="Verdana,sans-serif" font-size="11" '
        'text-anchor="middle"><text x="84" y="18">statement coverage</text>'
        f'<text x="202" y="18">{percent}</text></g></svg>\n')
    (output / 'coverage.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Talos statement coverage</title><body>'
        '<h1>Talos statement coverage</h1>'
        f'<p>{covered} of {total} statements covered ({percent}) across '
        f'{len(payload["files"])} package modules.</p>'
        f'<p>Source: <a href="https://github.com/autonomio/talos/tree/{commit}">{commit}</a>.</p>'
        f'<p><a href="{escape(run_url)}">Executed CI run</a> · '
        '<a href="coverage.json">Complete coverage report</a> · '
        '<a href="coverage-summary.json">Source and report digest</a>.</p>'
        '<p>This measures the complete framework suite and executable documentation. '
        'It is statement coverage, independent of branch coverage and the core-only floor.</p>'
        '</body></html>\n')
    return summary


def main() -> None:
    """Generate durable coverage files for the checked documentation build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--run-url', required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.report, args.output, args.commit, args.run_url), indent=2))


if __name__ == '__main__':
    main()
