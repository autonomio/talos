from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_issue_fetch_is_non_nullable_and_gate_has_no_none_guard() -> None:
    source = (REPO_ROOT / 'governance/cc_gate.py').read_text(encoding='utf-8')
    start = source.index('def fetch_issue(repo: str, number: int) -> dict[str, object]:')
    end = source.index('def find_closing_references(body: str) -> list[int]:')
    block = source[start:end]

    assert '| None' not in block
    assert 'return None' not in block
    assert 'raise SystemExit(2)' in block
    assert 'if issue_title is not None' not in source


def _load_cc_gate():
    import importlib.util
    spec = importlib.util.spec_from_file_location('cc_gate', REPO_ROOT / 'governance/cc_gate.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_attribution_hit_flags_ai_attribution() -> None:
    cc = _load_cc_gate()
    assert cc.attribution_hit('Co-Authored-By: Claude <noreply@anthropic.com>') is not None
    assert cc.attribution_hit('feat: generated with Copilot assistance') is not None
    assert cc.attribution_hit('chore: 🤖 generated with Claude') is not None
    assert cc.attribution_hit('refactor: address anthropic feedback') is not None
    # AI-qualified forms of the narrowed tokens stay flagged.
    assert cc.attribution_hit('docs: notes from a google-gemini session') is not None
    assert cc.attribution_hit('docs: gemini-pro prompt notes') is not None
    assert cc.attribution_hit('feat: cursor-ai integration pass') is not None
    assert cc.attribution_hit('chore: llm-generated summary') is not None
    assert cc.attribution_hit('Co-authored-by: GPT <bot@example.com>') is not None


def test_attribution_hit_passes_clean_messages() -> None:
    cc = _load_cc_gate()
    assert cc.attribution_hit('feat: add the maker-fill evaluator') is None
    assert cc.attribution_hit('docs: clarify the rollout runbook') is None
    assert cc.attribution_hit('fix: bound the version read') is None
    # The repo's own governance files and the Copilot review feature are
    # exempt -- they are filenames and a feature, not authorship.
    assert cc.attribution_hit('docs: retitle CLAUDE.md and point AGENTS.md at it') is None
    assert cc.attribution_hit('feat: update copilot-instructions.md and require Copilot review') is None
    # Domain vocabulary that names no AI assistant must pass: the bare
    # tokens are narrowed to AI-qualified forms, `api`/`code` are not
    # Gemini qualifiers, and the co-author alternation is word-bounded.
    assert cc.attribution_hit('feat: add Gemini exchange connector') is None
    assert cc.attribution_hit('feat: add Gemini API client for the Gemini exchange') is None
    assert cc.attribution_hit('fix: reuse the database cursor between batches') is None
    assert cc.attribution_hit('docs: describe artifacts generated with the manifest runner') is None
    assert cc.attribution_hit('Co-authored-by: Bill Hillman <hillman@example.com>') is None


def test_find_closing_references_deduplicates_in_order() -> None:
    cc = _load_cc_gate()
    assert cc.find_closing_references('Closes #9\nFixes #12\nCloses #9') == [9, 12]


def test_list_commits_fails_closed_on_unparseable_log_line(monkeypatch) -> None:
    import subprocess

    cc = _load_cc_gate()

    class FakeResult:
        returncode = 0
        stdout = 'deadbeef only-two-fields\n'
        stderr = ''

    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: FakeResult())
    with pytest.raises(SystemExit) as excinfo:
        cc.list_commits('base', 'head')
    assert excinfo.value.code == 2


def test_gate_checks_every_reference_and_exempts_only_prds(monkeypatch) -> None:
    cc = _load_cc_gate()
    issues = {
        9: {'title': 'bad slice title without cc form', 'labels': ['slice']},
        12: {'title': 'PRD: also not cc form, and exempt', 'labels': ['planning']},
        15: {'title': 'ordinary issue, also not cc form', 'labels': ['bug']},
    }
    monkeypatch.setattr(cc, 'fetch_issue', lambda _repo, number: issues[number])
    monkeypatch.setattr(cc, 'list_commits', lambda _base, _head: [])
    monkeypatch.setattr(cc, 'list_commit_messages', lambda _base, _head: [])

    failures = cc.gate(
        'feat: valid title',
        'Closes #9\nCloses #12\nCloses #15',
        'base',
        'head',
        'Autonomio/new-repository-template',
    )

    assert any('#9' in failure for failure in failures)
    assert any('#15' in failure for failure in failures)
    assert not any('#12' in failure for failure in failures)
