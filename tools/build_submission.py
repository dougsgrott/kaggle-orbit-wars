#!/usr/bin/env python3
"""
Build a single self-contained `submission.py` for the Kaggle Orbit Wars runner.

Kaggle executes a submission as one isolated file — there is no `src/` package on
its path — but our agent is split across `src/utils.py`, `src/features.py`, and
the `roi_greedy` brain in `src/agents/`, wired together by intra-package imports.
This packager flattens exactly the pieces the agent needs into one file with no
`src.` / relative imports, so the source of truth stays in `src/` and the
submission is regenerated on demand (this is the AG3 packaging deliverable).

What it does:
  * Reads the modules listed in SECTIONS, in order.
  * For an "ALL" section: emits the whole module verbatim minus its docstring and
    its import statements — preserving every comment (incl. the `# Variants:`
    notes that document the physics divergences).
  * For a name-list section: emits only those top-level defs (here, just
    `find_target_via_ray` — so numpy, used elsewhere in features.py, is dropped).
  * Hoists a single deduped import header (`from __future__`, stdlib only;
    intra-package and numpy imports are stripped).
  * Appends the top-level `agent(obs, config=None)` Kaggle entry point.

Then it SELF-VERIFIES the result in a subprocess run from a temp dir where `src`
is NOT importable — so a stray `from src ...` fails the build here instead of
silently on Kaggle — and plays one synthetic turn to confirm `agent()` returns a
legal-looking move list.

Usage:  python tools/build_submission.py [-o submission.py]
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parents[1]

# (module path relative to repo root, "ALL" | [top-level names to keep]).
# Order matters: definitions are emitted top to bottom.
SECTIONS: List[Tuple[str, Union[str, List[str]]]] = [
    ("src/utils.py", "ALL"),
    ("src/features.py", ["find_target_via_ray"]),
    # roi_greedy supplies the shared helpers (SHIP_BUFFER, _field,
    # _inbound_threats, defense_reserve); roi_greedy_predict is the DEFAULT brain
    # and is flattened LAST so its `plan_turn` is the one the entry point calls
    # (it shadows roi_greedy's identically-named `plan_turn`). Keep the DEFAULT
    # brain's module last here when promoting a new brain.
    ("src/agents/roi_greedy.py", "ALL"),
    ("src/agents/roi_greedy_predict.py", "ALL"),
]

# The brain the entry point forwards to: the LAST-defined `plan_turn` above
# (the DEFAULT brain).
ENTRY_BODY = "plan_turn(obs, config)"

# Imports we never carry into the flattened file.
_DROP_IMPORT_MODULES = {"numpy"}


def _is_intra_package(node: ast.ImportFrom) -> bool:
    """A relative import (level > 0) or an absolute one into our own package."""
    if node.level and node.level > 0:
        return True
    mod = (node.module or "").split(".")[0]
    return mod in {"src", "utils", "features", "agents"}


def _collect_imports(
    tree: ast.AST, plain: Set[str], from_imports: Dict[str, Set[str]]
) -> None:
    """Gather external stdlib imports to hoist; skip __future__, intra-package,
    and dropped (numpy) imports."""
    for node in tree.body:  # top level only
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in _DROP_IMPORT_MODULES:
                    plain.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__" or _is_intra_package(node):
                continue
            if (node.module or "").split(".")[0] in _DROP_IMPORT_MODULES:
                continue
            names = from_imports.setdefault(node.module, set())
            names.update(alias.name for alias in node.names)


def _emit_all(source: str, tree: ast.Module) -> str:
    """Whole module verbatim, minus its docstring and import statements."""
    lines = source.splitlines()
    drop: Set[int] = set()  # 1-based line numbers to remove

    body = tree.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        drop.update(range(body[0].lineno, body[0].end_lineno + 1))
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            drop.update(range(node.lineno, node.end_lineno + 1))

    kept = [ln for i, ln in enumerate(lines, start=1) if i not in drop]
    return "\n".join(kept).strip("\n")


def _emit_names(source: str, tree: ast.Module, names: List[str]) -> str:
    """Only the named top-level defs, each with any immediately-preceding comment
    block and decorators, verbatim."""
    lines = source.splitlines()
    wanted = set(names)
    blocks: List[str] = []
    found: Set[str] = set()

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name not in wanted:
            continue
        found.add(node.name)
        start = node.lineno
        if node.decorator_list:
            start = min(start, min(d.lineno for d in node.decorator_list))
        # Absorb a contiguous comment block sitting directly above the def.
        i = start - 1  # 1-based line just above `start`
        while i >= 1 and lines[i - 1].lstrip().startswith("#"):
            start = i
            i -= 1
        blocks.append("\n".join(lines[start - 1 : node.end_lineno]))

    missing = wanted - found
    if missing:
        raise SystemExit(f"build_submission: names not found: {sorted(missing)}")
    return "\n\n".join(blocks)


def _render_imports(plain: Set[str], from_imports: Dict[str, Set[str]]) -> str:
    out: List[str] = ["from __future__ import annotations", ""]
    for mod in sorted(plain):
        out.append(f"import {mod}")
    for mod in sorted(from_imports):
        names = ", ".join(sorted(from_imports[mod]))
        out.append(f"from {mod} import {names}")
    return "\n".join(out)


def build(out_path: Path) -> str:
    plain: Set[str] = set()
    from_imports: Dict[str, Set[str]] = {}
    bodies: List[str] = []

    for rel, what in SECTIONS:
        source = (REPO_ROOT / rel).read_text(encoding="utf-8")
        tree = ast.parse(source)
        _collect_imports(tree, plain, from_imports)
        if what == "ALL":
            body = _emit_all(source, tree)
        else:
            body = _emit_names(source, tree, list(what))
        bodies.append(f"# ===== from {rel} =====\n{body}")

    header = (
        '"""\n'
        "GENERATED — do not edit by hand.\n\n"
        "Self-contained Kaggle Orbit Wars submission, flattened from src/ by\n"
        "tools/build_submission.py. Edit the sources in src/ and regenerate:\n"
        "    python tools/build_submission.py\n\n"
        "Provenance: " + ", ".join(rel for rel, _ in SECTIONS) + "\n"
        '"""'
    )
    entry = (
        "# ===== Kaggle entry point =====\n"
        "def agent(obs, config=None):\n"
        f"    return {ENTRY_BODY}\n"
    )

    text = "\n\n\n".join(
        [header, _render_imports(plain, from_imports), *bodies, entry]
    ).rstrip() + "\n"

    out_path.write_text(text, encoding="utf-8")
    return text


def _static_check(text: str) -> None:
    """Fast structural guard: no forbidden import *statements* survived the
    flatten. Parse the output and inspect real import nodes, so prose/comments
    that merely mention "from src" don't trip it."""
    tree = ast.parse(text)
    bad: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad += [
                a.name for a in node.names
                if a.name.split(".")[0] in _DROP_IMPORT_MODULES | {"src"}
            ]
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if (node.level and node.level > 0) or root in (
                _DROP_IMPORT_MODULES | {"src", "utils", "features", "agents"}
            ):
                bad.append(f"from {'.' * (node.level or 0)}{node.module or ''}")
    if bad:
        raise SystemExit(f"build_submission: forbidden import(s) survived: {bad}")


def _isolation_check(out_path: Path) -> None:
    """Import the generated file and play one synthetic turn from a temp dir where
    `src` is NOT importable — the real proof it has no hidden src dependency."""
    probe = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('submission', r'{out_path}')\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "obs = {'player': 0,\n"
        "       'planets': [[0,0,10.0,10.0,1.0,80,1],[1,-1,40.0,10.0,1.0,5,1]],\n"
        "       'fleets': []}\n"
        "moves = m.agent(obs)\n"
        "assert isinstance(moves, list) and moves and moves[0][0] == 0, moves\n"
        "print('OK', moves)\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        # cwd=tmp and no PYTHONPATH => sys.path has no repo root, so `import src`
        # would fail here. Surfacing that now beats discovering it on Kaggle.
        res = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=tmp,
            env={"PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
        )
    if res.returncode != 0:
        raise SystemExit(
            "build_submission: isolation check FAILED — the generated file does "
            "not run standalone:\n" + (res.stderr or res.stdout)
        )
    print("isolation check:", res.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-o", "--out", default=str(REPO_ROOT / "submission.py"),
        help="output path (default: repo-root submission.py)",
    )
    args = ap.parse_args()
    out_path = Path(args.out).resolve()

    text = build(out_path)
    _static_check(text)
    _isolation_check(out_path)
    print(f"wrote {out_path} ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
