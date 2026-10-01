"""Contract tests for the package audit's dependency-bound rule.

The bound rule is the one that catches real breakage: an unbounded dependency
lets a resolver change what ships between two builds of the same version. It
is worth its own test because the first implementation reported the exact
class of dependency it exists to catch as compliant.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

from package_audit import unbounded


def _bounded(spec: str) -> bool:
    """Whether the audit itself considers this spec bounded."""
    return unbounded([spec]) == []


def test_ranges_and_exact_pins_count_as_bounded() -> None:
    """Verify both bound forms are accepted."""
    assert _bounded('pkg>=1,<2')
    assert _bounded('pkg>=1.2.3, <2')
    assert _bounded('pkg==1.2.3')


def test_environment_markers_do_not_supply_the_bound() -> None:
    """Verify a marker's own `==` cannot satisfy the bound rule.

    Matching over the whole requirement string let
    `pkg; python_version=="3.12"` pass: the `==` inside the marker looked like
    an exact pin. That is precisely an unbounded dependency, and reporting it
    as compliant defeated the rule.
    """
    assert not _bounded('pkg; python_version=="3.12"')
    assert not _bounded('pkg>=1; sys_platform=="linux"')
    assert not _bounded('pkg')
    assert not _bounded('pkg>=1')
    # A genuinely bounded spec keeps its bound when a marker follows it.
    assert _bounded('pkg>=1,<2; python_version<"3.11"')
