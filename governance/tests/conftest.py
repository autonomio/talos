"""Put `governance/` on `sys.path` so gate modules and tests can import the
shared `_common` helper by bare name — the same way the gates resolve it when
run as scripts (`python governance/check_*.py`), where `governance/` is the
script directory. Gates that are imported as `governance.<name>` in tests rely
on this too, since their internal `import _common` resolves against this path.
"""
from __future__ import annotations

import sys
from pathlib import Path

_GOVERNANCE = Path(__file__).resolve().parents[1]
if str(_GOVERNANCE) not in sys.path:
    sys.path.insert(0, str(_GOVERNANCE))
