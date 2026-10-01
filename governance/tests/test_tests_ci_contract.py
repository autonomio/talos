from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_WORKFLOW = REPO_ROOT / '.github/workflows/pr_checks_tests.yml'
EXPECTED_TEST_COMMAND = 'pytest -p governance.pytest_runtime tests/test_*.py -q --maxfail=1'


def test_pr_checks_tests_workflow_exists() -> None:
    assert TESTS_WORKFLOW.exists()


def test_pr_checks_tests_pins_python_and_runtime_suite_command() -> None:
    workflow = TESTS_WORKFLOW.read_text(encoding='utf-8')

    assert "python-version: '3.12'" in workflow
    assert 'python -m pip install --require-hashes -r requirements/ci/dev-env.txt' in workflow
    assert 'python -m pip install --require-hashes -r requirements/ci/runtime-env.txt' in workflow
    assert 'python -m pip install --require-hashes -r requirements/ci/build-tools.txt' in workflow
    assert 'python -m pip install --no-build-isolation --no-deps -e .' in workflow
    assert EXPECTED_TEST_COMMAND in workflow
    # No soft-fail pathway in any job that is a required check. The one
    # `continue-on-error` here is the artifact download in
    # `publish_coverage_comment`, which is informational and must not fail the
    # PR when the test job produced no artifact. Asserting per-job keeps that
    # allowance from silently widening to a gate.
    payload = yaml.safe_load(workflow)
    gating = {n: j for n, j in payload['jobs'].items() if n != 'publish_coverage_comment'}
    assert gating
    for name, job in gating.items():
        assert 'continue-on-error' not in job, name
        for step in job.get('steps', []):
            assert 'continue-on-error' not in step, (name, step.get('name'))
