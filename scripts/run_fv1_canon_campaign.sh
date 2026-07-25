#!/bin/bash
# NEW-CANON (fv1-bypass, linear s over (1,2)) full 2D recomputation in
# flow360_fv1 (author decision 2026-07-25). Sets run sequentially, each
# multiplexed over all 8 GPUs by run_sphere_campaign; flat plates last.
set -u
export SAAI_CAMPAIGN_ROOT=/home/qiqi/flexcompute/sa-ai/flow360_fv1
# NEW CANON: the bypass rides the environment into every solver child
# (converge_by_xtr inherits os.environ; the compiled default stays 0 so
# classical-SA runs are untouched).
export AI_FV1BYPASS=1
export AI_FV1_SWIDTH=1
cd /home/qiqi/flexcompute/sa-ai/flow360_ai
for SET in nlf eppler epp_sweep_l1 epp_sweep_levels nlf_neg eppler_ext; do
    echo "=== SET $SET $(date -u +%H:%M) ==="
    python3 run_sphere_campaign.py $SET 0,1,2,3,4,5,6,7
done
echo "=== FLAT PLATES $(date -u +%H:%M) ==="
python3 - <<'EOF'
import sys, os, threading
sys.path.insert(0, "/home/qiqi/flexcompute/flexfoil/rans")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/cfd")
sys.path.insert(0, "/home/qiqi/flexcompute/sa-ai/paper/repro/driver")
from rans.solve import run_solver
from run_continuation_ladders import canon_env, write_ai_constants, forces
NEW = "/home/qiqi/flexcompute/sa-ai/flow360_fv1"
TUS = ("Tu0040", "Tu0080", "Tu0160", "Tu0300", "Tu0600")

def one(tu, gpu):
    wd = f"{NEW}/flatplate_sphere_{tu}"
    env, find = canon_env()
    print(f"START flatplate {tu} gpu={gpu}", flush=True)
    run_solver(wd, find, env, gpu=gpu, timeout=14400)
    write_ai_constants(wd)
    print(f"DONE flatplate {tu}: {forces(wd)}", flush=True)

th = [threading.Thread(target=one, args=(tu, i)) for i, tu in enumerate(TUS)]
for t in th: t.start()
for t in th: t.join()
EOF
echo "FV1-CANON-CAMPAIGN-DONE"
