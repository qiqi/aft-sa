"""Disk janitor for the drag-crisis matrix (disk at 99% -- coordinator order).

Every pass, for each case dir under the matrix root:
  - mesh.msh / cylinder_ogrid.p3d: replace per-case COPIES with hardlinks to
    the template's (build-time inputs only; mesh.cgns is what the solver
    reads). Safe while the solver runs.
  - completed cases (summary.json present):
      * solver.log truncated to its last 200 KB (verbose GPU logs, ~26 MB/case);
        the resolved-constants echo is preserved to ai_constants_echo.log first.
      * volume*.pvtu deleted if any (matrix template has volumeOutput removed).
  - completed cases whose SUCCESSOR in the same chain is also complete (the
    ladder no longer needs this case's restart): restartOutput/, root
    restart*.dmp, restart.json deleted. Keeps: force CSVs, surface/slice
    outputs, summary.json, Flow360.json, logs (truncated).

Run: python dragcrisis_matrix_janitor.py [--root ...] [--interval 120]
Exits when all queues are done (queue_gpu*.log x4 + one final pass) or on
Ctrl-C.
"""
from __future__ import annotations

import argparse
import json
import os
import re as _re
import subprocess
import time
from pathlib import Path

RE_LIST = [6e4, 1e5, 1.5e5, 2e5, 2.5e5, 3e5, 3.5e5, 4e5, 5e5, 7e5, 1e6, 2e6]


def successor(name: str) -> str | None:
    m = _re.match(r"cyl_Re(\d+)_Tu([\d.]+)_(up|dn|cold)$", name)
    if not m:
        return None
    re_v, tu, d = float(m.group(1)), m.group(2), m.group(3)
    if d == "cold":
        return None                      # cold refs are independent
    seq = RE_LIST if d == "up" else RE_LIST[::-1]
    i = seq.index(re_v)
    if i + 1 >= len(seq):
        return None
    return f"cyl_Re{int(seq[i + 1])}_Tu{tu}_{d}"


def relink(case: Path, tmpl: Path, fname: str) -> int:
    f = case / fname
    t = tmpl / fname
    if f.is_file() and t.is_file() and f.stat().st_ino != t.stat().st_ino:
        sz = f.stat().st_size
        f.unlink()
        os.link(t, f)
        return sz
    return 0


def truncate_solver_log(case: Path) -> int:
    """Preserve the SA-AI constants echo, then truncate solver.log to its
    last 200 KB. Returns bytes freed. (Shared with the extension driver.)"""
    freed = 0
    slog = case / "solver.log"
    if slog.exists() and slog.stat().st_size > 300_000:
        echo = case / "ai_constants_echo.log"
        if not echo.exists():
            out = subprocess.run(
                ["grep", "-a", "-m1", "-A", "17",
                 "SA-AI transition constants", str(slog)],
                capture_output=True, text=True).stdout
            echo.write_text(out)
        sz = slog.stat().st_size
        data = slog.read_bytes()[-200_000:]
        slog.write_bytes(b"[janitor: truncated to last 200KB]\n" + data)
        freed = sz - slog.stat().st_size
    return freed


def purge_restarts(case: Path) -> int:
    """Delete a superseded case's restart files. Returns bytes freed.
    (Shared with the extension driver.)"""
    freed = 0
    ro = case / "restartOutput"
    if ro.is_dir():
        for f in ro.iterdir():
            freed += f.stat().st_size
            f.unlink()
        ro.rmdir()
    for f in list(case.glob("restart*")):
        if f.is_file():
            freed += f.stat().st_size
            f.unlink()
    return freed


def pass_once(root: Path) -> int:
    tmpl = root / "template_case"
    freed = 0
    for case in sorted(root.glob("cyl_Re*")):
        if not case.is_dir():
            continue
        for fname in ("mesh.msh", "cylinder_ogrid.p3d"):
            freed += relink(case, tmpl, fname)
        done = (case / "summary.json").exists()
        if not done:
            continue
        freed += truncate_solver_log(case)
        for v in case.glob("volume*"):
            if v.is_file():
                freed += v.stat().st_size
                v.unlink()
        succ = successor(case.name)
        if succ and (root / succ / "summary.json").exists():
            freed += purge_restarts(case)
    return freed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/local_data/qiqi/sa-ai/dragcrisis_matrix")
    ap.add_argument("--interval", type=int, default=120)
    args = ap.parse_args()
    root = Path(args.root)
    while True:
        freed = pass_once(root)
        df = os.statvfs(root)
        free_gb = df.f_bavail * df.f_frsize / 2 ** 30
        print(f"janitor: freed {freed / 2**20:.0f} MB this pass; "
              f"{free_gb:.1f} GB free", flush=True)
        all_done = all((root / f"queue_gpu{q}.log").exists() for q in range(4))
        if all_done:
            pass_once(root)
            print("janitor: all queues done; final pass complete", flush=True)
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
