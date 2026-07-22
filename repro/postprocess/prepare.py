"""Prepare a CFD results tree for the paper's figure/table generators.

The generators (``paper/regen_*.py``) read one flat directory of case dirs,
keyed by canonical case name, through the env var ``SAAI_CFD_ROOT``
(default: the shipped ``sa-ai/flow360_a3/``). Two preparation steps are
needed before the NLF/Eppler surface figures can be drawn from FRESH solves:

1. **Tree assembly** -- symlink every canonical case name into one root.
   Fresh results (e.g. ``repro/driver/out/<case>``) take precedence; any case
   not re-run falls back to the shipped ``flow360_a3`` entry, so a partial
   re-run still yields a complete tree.

2. **Derived slice fields** -- ``flow360/add_derived_to_slice.py`` must be run
   over the 48 airfoil cases (writes ``slice_with_derived_0.vtu`` next to each
   ``slice_centerSpan.pvtu``; required by regen_nlf_v2 / regen_eppler_v2).

Usage:
    python3 prepare.py --tree /path/to/tree [--fresh repro/driver/out] [--skip-derived]

Then: SAAI_CFD_ROOT=/path/to/tree python3 regenerate_cfd.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SAAI = Path(__file__).resolve().parent.parent.parent
SHIPPED = SAAI / "flow360_a3"
sys.path.insert(0, str(SAAI / "repro" / "driver"))
sys.path.insert(0, str(SAAI / "flow360"))


def build_tree(tree: Path, fresh_roots: list[Path]) -> int:
    """Symlink canonical case names into ``tree``; fresh roots win over shipped."""
    import cases
    tree.mkdir(parents=True, exist_ok=True)
    n_fresh = 0
    for name in cases.all_cases():
        target, fresh = None, False
        for root in fresh_roots:
            cand = root / name
            if (cand / "Flow360.json").exists():
                target, fresh = cand, True
                break
        if target is None and (SHIPPED / name).exists():
            target = SHIPPED / name
        if target is None:
            print(f"  MISSING {name} (not in fresh roots or shipped tree)")
            continue
        link = tree / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target.resolve())
        n_fresh += fresh
    # e^N reference caches (mfoil/xfoil pickles) the generators read from the root
    for aux_path in SHIPPED.glob("*.pkl"):
        link = tree / aux_path.name
        if not (link.is_symlink() or link.exists()):
            link.symlink_to(aux_path.resolve())
    print(f"tree at {tree}: {len(list(tree.iterdir()))} entries ({n_fresh} fresh)")
    return 0


def add_derived(tree: Path) -> None:
    """Run add_derived_to_slice over every airfoil case in the tree."""
    import add_derived_to_slice as A
    cases_ = A.find_cases(str(tree), "nlf0416_Re4M") + A.find_cases(str(tree), "eppler387_Re200k")
    ok = 0
    for c in cases_:
        p = os.path.join(c, "slice_centerSpan.pvtu")
        if not os.path.exists(p):
            print(f"  SKIP {os.path.basename(c)}: no slice_centerSpan.pvtu")
            continue
        s, msg = A.augment(p)
        ok += bool(s)
        if not s:
            print(f"  FAIL {os.path.basename(c)}: {msg}")
    print(f"derived slices: {ok}/{len(cases_)} augmented")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tree", required=True, help="tree directory to (re)build")
    ap.add_argument("--fresh", action="append", default=[],
                    help="root(s) of fresh driver output, highest priority first")
    ap.add_argument("--skip-derived", action="store_true")
    args = ap.parse_args(argv)
    tree = Path(args.tree)
    build_tree(tree, [Path(f) for f in args.fresh])
    if not args.skip_derived:
        add_derived(tree)


if __name__ == "__main__":
    main()
