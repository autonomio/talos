"""Pytest configuration for the package suite.

Emits a runtime profile when `TEST_RUNTIME_PROFILE` names a path, so
`governance/check_test_runtime.py` has something to check the suite against.
Writing it is opt-in: a local run should not litter the tree, and CI sets the
variable explicitly.
"""
from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Any

_STARTED: dict[str, float] = {}
_RECORDS: list[dict[str, Any]] = []
_SUITE_START: float | None = None


def pytest_sessionstart(session: object) -> None:
    """Record when the suite began."""
    global _SUITE_START
    _SUITE_START = time.perf_counter()


def pytest_runtest_logstart(nodeid: str, location: object) -> None:
    """Record when one test began."""
    _STARTED[nodeid] = time.perf_counter()


def pytest_runtest_logfinish(nodeid: str, location: object) -> None:
    """Record how long one test took."""
    started = _STARTED.pop(nodeid, None)
    if started is not None:
        _RECORDS.append({'name': nodeid, 'duration': time.perf_counter() - started})


def pytest_sessionfinish(session: object, exitstatus: int) -> None:
    """Write the runtime profile when a destination is configured."""
    destination = os.environ.get('TEST_RUNTIME_PROFILE')
    if not destination or _SUITE_START is None:
        return
    path = pathlib.Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                'total_seconds': time.perf_counter() - _SUITE_START,
                'tests': sorted(_RECORDS, key=lambda row: -float(row['duration'])),
            },
            indent=2,
        )
        + '\n',
        encoding='utf-8',
    )
