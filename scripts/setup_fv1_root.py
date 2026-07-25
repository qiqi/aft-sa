"""Stage the NEW-CANON (fv1-bypass, linear s over (1,2)) 2D campaign root:
/local_data/qiqi/sa-ai/flow360_fv1, symlinked at sa-ai/flow360_fv1. Every
case dir is cloned (mesh hardlinked, Flow360.json copied, cold) from its
old-canon counterpart in flow360_fr, so the old canon stays intact for
provenance and the paper flips roots via SAAI_CFD_ROOT when adopted.

Cases: NLF 24 + negative pair (L2 both fams); Eppler 24 + extension alphas
(-2, 1, 3, 4, 6, 8.5 on both L2s); the Re sweep 24; flat plate 5.

  python3 setup_fv1_root.py
"""
import os, sys, json, shutil

sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/cfd")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/driver")
from run_continuation_ladders import FR

NEW = "/home/qiqi/flexcompute/sa-ai/flow360_fv1"
NEW_REAL = "/local_data/qiqi/sa-ai/flow360_fv1"
os.makedirs(NEW_REAL, exist_ok=True)
if not os.path.islink(NEW) and not os.path.exists(NEW):
    os.symlink(NEW_REAL, NEW)

import run_continuation_ladders as RCL
from run_continuation_ladders import OUT_NAMES, OUT_PAT


def _clone(src, dst):
    """Local clone: the root is already a /local_data symlink, so no
    per-case symlink layer -- hardlink big inputs, copy small ones."""
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    for f in os.listdir(src):
        if f in OUT_NAMES or f.endswith(OUT_PAT):
            continue
        sp = os.path.join(src, f)
        if not os.path.isfile(sp):
            continue
        if os.path.getsize(sp) > 10e6:
            try:
                os.link(os.path.realpath(sp), os.path.join(dst, f))
            except OSError:
                shutil.copy(sp, os.path.join(dst, f))
        else:
            shutil.copy(sp, os.path.join(dst, f))

JOBS = []      # (new-case name, source case, alphaAngle or None)
for m in ("cav", "str"):
    for L in ("L0", "L1", "L2"):
        for a in (0, 4, 9, 15):
            JOBS.append((f"{m}{L}prop_nlf0416_Re4M_a{a}",
                         f"{m}{L}prop_nlf0416_Re4M_a{a}", None))
        for a in (0, 2, 5, 7):
            JOBS.append((f"{m}{L}prop_eppler387_Re200k_a{a}",
                         f"{m}{L}prop_eppler387_Re200k_a{a}", None))
    for L in ("L0", "L1", "L2"):
        for Rk in (60, 100, 300, 460):
            # the L1 sweep rows live under legacy names in flow360_fr
            src = (f"sweep_{m}{L}_Re{Rk}k_a5" if L != "L1" else
                   (f"sweep_Re{Rk}k_a5" if m == "cav" else f"sweep_str_Re{Rk}k_a5"))
            JOBS.append((f"sweep_{m}{L}_Re{Rk}k_a5", src, None))
for m in ("cav", "str"):
    JOBS.append((f"{m}L2prop_nlf0416_Re4M_am4", f"{m}L2prop_nlf0416_Re4M_a4", -4.0))
    JOBS.append((f"{m}L2prop_nlf0416_Re4M_am8", f"{m}L2prop_nlf0416_Re4M_a4", -8.0))
    for a, tag in ((-2.0, "am2"), (1.0, "a1"), (3.0, "a3"), (4.0, "a4x"),
                   (6.0, "a6"), (8.5, "a8p5")):
        # a4x avoids colliding with the benchmark a4 (which is NOT a
        # benchmark incidence for the Eppler; it IS an extension point)
        JOBS.append((f"{m}L2prop_eppler387_Re200k_{tag}",
                     f"{m}L2prop_eppler387_Re200k_a5", a))
for tu in ("Tu0040", "Tu0080", "Tu0160", "Tu0300", "Tu0600"):
    JOBS.append((f"flatplate_sphere_{tu}", f"flatplate_sphere_{tu}", None))

missing = [s for _, s, _ in JOBS if not os.path.isdir(f"{FR}/{s}")]
if missing:
    print("MISSING sources:", missing)
    raise SystemExit(1)

for name, src, alpha in JOBS:
    dst = f"{NEW}/{name}"
    if os.path.isdir(dst) and os.path.exists(f"{dst}/Flow360.json"):
        print("skip (exists)", name)
        continue
    _clone(f"{FR}/{src}", dst)
    j = json.load(open(f"{dst}/Flow360.json"))
    j['runControl']['restart'] = False
    if alpha is not None:
        j['freestream']['alphaAngle'] = alpha
    json.dump(j, open(f"{dst}/Flow360.json", 'w'), indent=4)
    for f in ("restart.json", "restart_rank_1_of_1.dmp"):
        p = f"{dst}/{f}"
        os.path.exists(p) and os.remove(p)
    print("staged", name, f"(alpha={alpha})" if alpha is not None else "")
print(f"STAGED {len(JOBS)} cases in {NEW}")
