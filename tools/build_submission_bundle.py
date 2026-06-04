#!/usr/bin/env python3
"""
Build a self-contained Kaggle submission for a brain that needs the WHOLE `src/`
package at runtime (e.g. `lookahead`, which pulls in the WorldModel + every
candidate brain + utils/features/arena). The flat packager
(`tools/build_submission.py`) can only flatten a self-contained single brain; the
search brains are an interdependent web (colliding `plan_turn`/`_field`, module-
qualified `worldmodel.step`, the runtime `kaggle_environments` import), so we
bundle instead.

Approach — **zip-bootstrap**: embed the `src/` package as a base64'd zip; at
import time the submission extracts it to a temp dir, puts it on `sys.path`, and
imports normally. Every import (relative, `kaggle_environments`, `numpy`) resolves
unchanged — zero flattening, zero collisions. The agent is `REGISTRY[DEFAULT]`.

This is the **option (A)** packaging: the brain imports the interpreter at runtime
(via `src.worldmodel`). Risks (accepted, to be measured on Kaggle): writing to a
temp dir must be permitted, and the env must be importable from inside a running
agent (verified locally via `env.run`).

Usage:  python tools/build_submission_bundle.py [-o submission.py]
"""
from __future__ import annotations

import argparse
import base64
import io
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

# Skip test code, caches, and bulky dev-only modules not needed at play time.
_SKIP_DIRS = {"__pycache__", "tests"}
_SKIP_SUFFIXES = {".pyc", ".pyo"}


def _zip_src() -> bytes:
    """Zip the `src/` package (sans tests/caches) into an in-memory archive,
    arcnames rooted so that `import src...` works after extraction."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(SRC.rglob("*")):
            if path.is_dir():
                continue
            rel = path.relative_to(REPO_ROOT)  # e.g. src/agents/lookahead.py
            if any(part in _SKIP_DIRS for part in rel.parts):
                continue
            if path.suffix in _SKIP_SUFFIXES:
                continue
            zf.write(path, arcname=str(rel))
    return buf.getvalue()


_BOOTSTRAP = '''"""
GENERATED — do not edit by hand. Self-contained Kaggle Orbit Wars submission
(zip-bootstrap of the src/ package). Regenerate:
    python tools/build_submission_bundle.py

This is the "option (A)" packaging: the agent imports kaggle_environments'
interpreter at runtime (via src.worldmodel) to do greedy lookahead search.
Bundled brain: {default}.
"""
import base64
import hashlib
import io
import os
import sys
import tempfile
import zipfile

_SRC_ZIP_B64 = "{b64}"


def _bootstrap():
    """Extract the bundled src/ package to a content-addressed temp dir and put it
    on sys.path. Keying the dir on a hash of the payload means a changed bundle
    re-extracts (no stale src) and distinct bundles never collide."""
    data = base64.b64decode(_SRC_ZIP_B64)
    digest = hashlib.sha1(data).hexdigest()[:12]
    target = os.path.join(tempfile.gettempdir(), "orbit_wars_sub_" + digest)
    marker = os.path.join(target, ".extracted")
    if not os.path.exists(marker):
        os.makedirs(target, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(target)
        open(marker, "w").close()
    if target not in sys.path:
        sys.path.insert(0, target)


_bootstrap()

from src.agents import REGISTRY, DEFAULT  # noqa: E402

_PLAN = REGISTRY[DEFAULT]


def agent(obs, config=None):
    return _PLAN(obs, config)
'''


def build(out_path: Path) -> str:
    b64 = base64.b64encode(_zip_src()).decode("ascii")
    default = (SRC / "agents" / "__init__.py").read_text()
    # surface which brain is bundled (best-effort, for the docstring only)
    dflt = "lookahead" if 'DEFAULT = "lookahead"' in default else "DEFAULT"
    text = _BOOTSTRAP.format(b64=b64, default=dflt)
    out_path.write_text(text, encoding="utf-8")
    return text


# Probe board (static) reused by the self-test.
_PROBE = (
    "{'player':0,"
    "'planets':[[0,0,90.0,90.0,1.0,80,1],[1,-1,60.0,90.0,1.0,5,1]],"
    "'fleets':[],'initial_planets':[[0,0,90.0,90.0,1.0,80,1],[1,-1,60.0,90.0,1.0,5,1]],"
    "'angular_velocity':0.0,'comets':[],'comet_planet_ids':[]}"
)


def _self_test(out_path: Path) -> None:
    """Prove self-containment + the option-(A) runtime path in a subprocess where
    the repo's src/ is NOT importable: (1) import the bundle, (2) call agent()
    directly, (3) run it as a FILE-PATH agent through env.run (the Kaggle model,
    exercising the re-entrant interpreter import)."""
    probe = (
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('submission', r'{out_path}')\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        f"obs = {_PROBE}\n"
        "mv = m.agent(obs)\n"
        "assert isinstance(mv, list) and mv and mv[0][0] == 0, ('direct call bad', mv)\n"
        "from kaggle_environments import make\n"
        f"env = make('orbit_wars', configuration={{'seed':3001,'episodeSteps':40}}, debug=False)\n"
        f"env.run([r'{out_path}', 'starter'])\n"
        "st = env.steps[-1]\n"
        "stats = [s.status for s in st]\n"
        "assert all(s != 'ERROR' and s != 'INVALID' for s in stats), ('env.run faulted', stats)\n"
        "print('OK direct=%r env_statuses=%r' % (mv, stats))\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        res = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=tmp,                       # no repo on cwd
            env={"PATH": "/usr/bin:/bin"},  # no PYTHONPATH -> src only via the bundle
            capture_output=True, text=True,
        )
    ok = [ln for ln in res.stdout.splitlines() if ln.startswith("OK ")]
    if res.returncode != 0 or not ok:
        raise SystemExit(
            "bundle self-test FAILED:\n" + (res.stderr or res.stdout)[-3000:]
        )
    print("self-test:", ok[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default=str(REPO_ROOT / "submission.py"))
    args = ap.parse_args()
    out_path = Path(args.out).resolve()
    text = build(out_path)
    _self_test(out_path)
    print(f"wrote {out_path} ({len(text.encode()) // 1024} KiB)")


if __name__ == "__main__":
    main()
