from __future__ import annotations

import ast

from governance import fail_loud_gate


def _counts(source: str) -> dict[str, int]:
    return fail_loud_gate._count_in_tree(ast.parse(source))


def test_detects_empty_handler_categories() -> None:
    counts = _counts(
        'def f():\n'
        '    try:\n'
        '        work()\n'
        '    except:\n'
        '        pass\n'
        '    try:\n'
        '        work()\n'
        '    except ValueError:\n'
        '        ...\n'
        '    try:\n'
        '        work()\n'
        '    except RuntimeError:\n'
        '        return None\n'
        '    while True:\n'
        '        try:\n'
        '            work()\n'
        '        except OSError:\n'
        '            continue\n'
    )
    assert counts['bare_except'] == 1
    assert counts['empty_pass'] == 1
    assert counts['empty_ellipsis'] == 1
    assert counts['empty_return_none'] == 1
    assert counts['empty_continue_break'] == 1


def test_detects_contextlib_suppress_alias_chains() -> None:
    counts = _counts(
        'import contextlib as c\n'
        'mod = c\n'
        'sup = mod.suppress\n'
        'sup2 = sup\n'
        'def f():\n'
        '    with mod.suppress(ValueError):\n'
        '        work()\n'
        '    with sup2(RuntimeError):\n'
        '        work()\n'
    )
    assert counts['contextlib_suppress'] == 2


def test_detects_direct_contextlib_suppress_import_alias() -> None:
    counts = _counts(
        'from contextlib import suppress as silence\n'
        'def f():\n'
        '    with silence(ValueError):\n'
        '        work()\n'
    )
    assert counts['contextlib_suppress'] == 1


def test_function_local_assignment_alias_does_not_leak_to_module_scope() -> None:
    counts = _counts(
        'import contextlib\n'
        'def configure():\n'
        '    local_alias = contextlib\n'
        'def f():\n'
        '    with local_alias.suppress(ValueError):\n'
        '        work()\n'
    )
    assert counts['contextlib_suppress'] == 0


def test_detects_errors_ignore_kwarg() -> None:
    counts = _counts(
        'def f(data):\n'
        '    return data.decode("utf-8", errors="ignore")\n'
    )
    assert counts['errors_ignore_kwarg'] == 1
