"""``producer_port`` brain — the full Producer, vendored verbatim, as our champion base.

This is the entire Producer ("The Producer", slawekbiel) running inside our stack: it
calls the vendored runtime (`vendor/producer_runtime.py` + `vendor/orbit_lite/`) — the
same engine that scored **1222** on the public LB (sub_07), ~140 above our previous
best config-port (sub_06 = 1080). We replicate it faithfully first (battle-validated vs
the pristine original), then layer genuine contributions on top. See `vendor/NOTICE.md`.

It uses its OWN module-global runtime (separate from the `sota` opponent at
`src/opponents/producer.py`), so the two can battle in one process without sharing the
rolling fleet-cache. Requires torch. Returns our `[from_planet_id, angle, num_ships]`
Shots (the runtime's `sparse_action_row_to_moves` already emits that shape).

Public API:
    plan_turn(obs, config=None) -> list[list]
"""
from __future__ import annotations

import os
import sys
from typing import List

# Put the vendored package dir on the path so `import producer_runtime` (and the
# sibling `orbit_lite` it imports) resolve. Repo root is three dirs up from here.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VENDOR = os.path.join(_REPO, "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from producer_runtime import agent as _producer_agent  # noqa: E402  (vendored base)


def plan_turn(obs, config=None) -> List[list]:
    """Return this turn's Shots from the vendored Producer runtime.

    The vendored `agent` is the notebook's verbatim single-arg entry; `config`
    (the env configuration our harness may pass) is accepted here and ignored."""
    return _producer_agent(obs)
